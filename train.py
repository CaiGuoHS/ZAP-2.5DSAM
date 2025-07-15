import os

import argparse
import logging

import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import DiceLoss

from segment_anything import sam_model_registry
from modeling.image_encoder_zap import ImageEncoderViT
from modeling.mask_decoder_zap import MaskDecoder
from segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.modeling.prompt_encoder import PromptEncoder

from utils.datasets import load_data_volume
from utils.util import save_checkpoint
from utils.util import setup_logger

torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_points", default=3, type=int)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--beta", default=0.8, type=float)
    parser.add_argument("--new_window_size", default=16, type=float)

    parser.add_argument(
        "--data", default="colon", type=str, choices=["kits", "pancreas", "lits", "colon"]
    )
    parser.add_argument(
        "--data_dir",
        default="datasets/colon",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        default="ckpts",
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=(128, 128, 128),
        nargs='+', type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max_epoch", default=200, type=int)
    parser.add_argument("--num_worker", default=4, type=int)

    args = parser.parse_args()
    device = args.device
    save_path = os.path.join(args.save_dir, args.data, "_".join(["zap_2.5dsam", str(args.num_points), str(args.alpha), str(args.beta), str(args.new_window_size)]))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setup_logger(logger_name="train", root=save_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))
    setup_logger(logger_name="train_loss", root=save_path, screen=True, tofile=True)
    logger_loss = logging.getLogger(f"train_loss")

    train_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_dir,
        batch_size=1,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker = args.num_worker
    )
    val_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_dir,
        batch_size=1,
        augmentation=False,
        split="val",
        deterministic=True,
        rand_crop_spatial_size=args.rand_crop_size,
        do_val_crop=True,
        num_worker = args.num_worker
    )

    sam = sam_model_registry["vit_b"](checkpoint="ckpts/sam_vit_b_01ec64.pth")

    img_encoder = ImageEncoderViT(
        depth = 12,
        embed_dim = 768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=256,
        embedding_size = 32,
        new_window_size=args.new_window_size,
    )

    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(32, 32),
        input_image_size=(512, 512),
        mask_in_chans=16,
    )

    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
    )

    img_encoder.load_state_dict(sam.image_encoder.state_dict(), strict=False)
    prompt_encoder.load_state_dict(sam.prompt_encoder.state_dict())

    mask_decoder.iou_token.weight = sam.mask_decoder.iou_token.weight
    mask_decoder.mask_tokens.weight = sam.mask_decoder.mask_tokens.weight
    mask_decoder.transformer.load_state_dict(sam.mask_decoder.transformer.state_dict())
    mask_decoder.output_upscaling.load_state_dict(sam.mask_decoder.output_upscaling.state_dict())
    mask_decoder.output_hypernetworks_mlp.load_state_dict(sam.mask_decoder.output_hypernetworks_mlps[0].state_dict())

    del sam

    for param in img_encoder.parameters():
        param.requires_grad = False

    for param in prompt_encoder.parameters():
        param.requires_grad = False

    img_encoder.to(device)
    prompt_encoder.to(device)
    mask_decoder.to(device)
    img_encoder.update_pos_embed()

    decoder_opt = torch.optim.AdamW([i for i in mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0.00001)
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.1, total_iters=args.max_epoch)

    seg_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    best_loss = np.inf
    patch_size = args.rand_crop_size[0]
    sparse_embeddings_n, dense_embeddings = prompt_encoder(
        points=None,
        boxes=None,
        masks=None,
    )
    image_pe = prompt_encoder.get_dense_pe()

    img_encoder.eval()
    prompt_encoder.eval()
    mask_decoder.eval()

    for epoch_num in range(args.max_epoch):
        loss_summary = []
        mask_decoder.train()
        for idx, (img, seg, spacing, path, affine) in enumerate(train_data):
            print('seg: ', seg.sum(), path, img.shape, seg.shape)

            with torch.no_grad():
                img = img.to(device)
                l = len(np.where(seg == 1)[0])
                if l >= args.num_points:
                    sample = np.random.choice(np.arange(l), np.random.randint(1, args.num_points+1), replace=False)
                    num_samples = len(sample)
                    points_x = np.where(seg == 1)[-1][sample]
                    points_y = np.where(seg == 1)[-2][sample]
                    points_z = np.where(seg == 1)[-3][sample]
                    x = points_x * 512 / patch_size
                    y = points_y * 512 / patch_size
                    z = points_z * 512 / patch_size
                    points_d = [[x[i], y[i]] for i in range(num_samples)]
                    points_h = [[z[i], x[i]] for i in range(num_samples)]
                    points_w = [[y[i], z[i]] for i in range(num_samples)]

                    points_d_torch = torch.tensor(points_d, dtype=torch.float32, device=device).unsqueeze(0)
                    points_w_torch = torch.tensor(points_w, dtype=torch.float32, device=device).unsqueeze(0)
                    points_h_torch = torch.tensor(points_h, dtype=torch.float32, device=device).unsqueeze(0)
                    labels = [1 for _ in range(num_samples)]
                    labels_torch = torch.tensor(labels, dtype=torch.int32, device=device).unsqueeze(0)

                    sparse_embeddings_d, _ = prompt_encoder(
                        points=(points_d_torch, labels_torch),
                        boxes=None,
                        masks=None,
                    )
                    sparse_embeddings_h, _ = prompt_encoder(
                        points=(points_h_torch, labels_torch),
                        boxes=None,
                        masks=None,
                    )
                    sparse_embeddings_w, _ = prompt_encoder(
                        points=(points_w_torch, labels_torch),
                        boxes=None,
                        masks=None,
                    )
                else:
                    sparse_embeddings_d = sparse_embeddings_n
                    sparse_embeddings_h = sparse_embeddings_n
                    sparse_embeddings_w = sparse_embeddings_n

                # axial plane
                input_batch_d = F.interpolate(img[0].permute(1, 0, 2, 3), scale_factor=512/patch_size, mode='bilinear')
                image_embeddings_d = img_encoder(input_batch_d)
                # coronal plane
                input_batch_h = F.interpolate(img[0].permute(2, 0, 3, 1), scale_factor=512/patch_size, mode='bilinear')
                image_embeddings_h = img_encoder(input_batch_h)
                # sagittal plane
                input_batch_w = F.interpolate(img[0].permute(3, 0, 1, 2), scale_factor=512/patch_size, mode='bilinear')
                image_embeddings_w = img_encoder(input_batch_w)

            masks = mask_decoder(
                image_embeddings=(image_embeddings_d, image_embeddings_h, image_embeddings_w),
                image_pe=image_pe,
                sparse_prompt_embeddings=(sparse_embeddings_d, sparse_embeddings_h, sparse_embeddings_w),
                dense_prompt_embeddings=dense_embeddings,
            )

            seg = seg.to(device)
            loss = args.alpha * bce_loss(masks, seg.float()) + args.beta * seg_loss(masks, seg)
            decoder_opt.zero_grad()
            loss.backward()
            decoder_opt.step()
            loss_summary.append(loss.detach().cpu().numpy())
            logger.info('epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data)) + ": loss:" + str(loss_summary[-1].flatten()[0]))

        decoder_scheduler.step()
        logger.info("- Train metrics: " + str(np.mean(loss_summary)))
        logger_loss.info("Train " + str(np.mean(loss_summary)))

        mask_decoder.eval()
        with torch.no_grad():
            loss_summary = []
            for idx, (img, seg, spacing, path, affine) in enumerate(val_data):
                print('seg: ', seg.sum(), path, img.shape, seg.shape)
                img = img.to(device)
                l = len(np.where(seg == 1)[0])
                if l >= args.num_points:
                    sample = np.random.choice(np.arange(l), args.num_points, replace=False)
                    num_samples = len(sample)
                    points_x = np.where(seg == 1)[-1][sample]
                    points_y = np.where(seg == 1)[-2][sample]
                    points_z = np.where(seg == 1)[-3][sample]
                    x = points_x * 512 / patch_size
                    y = points_y * 512 / patch_size
                    z = points_z * 512 / patch_size
                    points_d = [[x[i], y[i]] for i in range(num_samples)]
                    points_h = [[z[i], x[i]] for i in range(num_samples)]
                    points_w = [[y[i], z[i]] for i in range(num_samples)]

                    points_d_torch = torch.tensor(points_d, dtype=torch.float32, device=device).unsqueeze(0)
                    points_w_torch = torch.tensor(points_w, dtype=torch.float32, device=device).unsqueeze(0)
                    points_h_torch = torch.tensor(points_h, dtype=torch.float32, device=device).unsqueeze(0)
                    labels = [1 for _ in range(num_samples)]
                    labels_torch = torch.tensor(labels, dtype=torch.int32, device=device).unsqueeze(0)

                    sparse_embeddings_d, _ = prompt_encoder(
                        points=(points_d_torch, labels_torch),
                        boxes=None,
                        masks=None,
                    )
                    sparse_embeddings_h, _ = prompt_encoder(
                        points=(points_h_torch, labels_torch),
                        boxes=None,
                        masks=None,
                    )
                    sparse_embeddings_w, _ = prompt_encoder(
                        points=(points_w_torch, labels_torch),
                        boxes=None,
                        masks=None,
                    )
                else:
                    sparse_embeddings_d = sparse_embeddings_n
                    sparse_embeddings_h = sparse_embeddings_n
                    sparse_embeddings_w = sparse_embeddings_n

                # axial plane
                input_batch_d = F.interpolate(img[0].permute(1, 0, 2, 3), scale_factor=512/patch_size, mode='bilinear')
                image_embeddings_d = img_encoder(input_batch_d)
                # coronal plane
                input_batch_h = F.interpolate(img[0].permute(2, 0, 3, 1), scale_factor=512/patch_size, mode='bilinear')
                image_embeddings_h = img_encoder(input_batch_h)
                # sagittal plane
                input_batch_w = F.interpolate(img[0].permute(3, 0, 1, 2), scale_factor=512/patch_size, mode='bilinear')
                image_embeddings_w = img_encoder(input_batch_w)

                masks = mask_decoder(
                    image_embeddings=(image_embeddings_d, image_embeddings_h, image_embeddings_w),
                    image_pe=image_pe,
                    sparse_prompt_embeddings=(sparse_embeddings_d, sparse_embeddings_h, sparse_embeddings_w),
                    dense_prompt_embeddings=dense_embeddings,
                )

                seg = seg.to(device)
                loss = seg_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                logger.info('epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(val_data)) + ": loss:" + str(loss_summary[-1].flatten()[0]))

        logger.info("- Val metrics: " + str(np.mean(loss_summary)))
        logger_loss.info("Val " + str(np.mean(loss_summary)))

        is_best = False
        if np.mean(loss_summary) < best_loss:
            best_loss = np.mean(loss_summary)
            is_best = True
            logger.info(f"Evolution best: {epoch_num}")

        save_checkpoint({
                         "decoder_dict": mask_decoder.state_dict(),
                         },
                        is_best=is_best,
                        checkpoint=save_path)
        logger.info("- Val metrics best: " + str(best_loss))


if __name__ == "__main__":
    main()