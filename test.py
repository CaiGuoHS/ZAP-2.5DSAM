import os

from monai.data import NibabelWriter

import argparse
import logging
import numpy as np
from functools import partial

import torch
import torch.nn.functional as F

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference

from modeling.image_encoder_zap import ImageEncoderViT as ImageEncoderViT
from modeling.mask_decoder_zap import MaskDecoder
from segment_anything import sam_model_registry
from segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.modeling.prompt_encoder import PromptEncoder

from utils.datasets import load_data_volume
from utils.util import model_predict
from utils.util import setup_logger
import utils.SurfaceDistance as SD

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_points", default=3, type=int)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--beta", default=0.8, type=float)
    parser.add_argument("--new_window_size", default=16, type=float)
    parser.add_argument("--num_times", default=1, type=int)
    parser.add_argument("--save_masks", default=True, type=bool)
    
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
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=4, type=int)
    parser.add_argument("--checkpoint", default="last", type=str)
    parser.add_argument("--tolerance", default=5, type=int)
    args = parser.parse_args()
    if args.checkpoint == "last":
        file = "last.pth.tar"
    else:
        file = "best.pth.tar"
    device = args.device

    save_path = os.path.join(args.save_dir, args.data, "_".join(["zap_2.5dsam", str(args.num_points), str(args.alpha), str(args.beta), str(args.new_window_size)]))
    setup_logger(logger_name=f"test{args.checkpoint}_{args.num_times}", root=save_path, screen=True, tofile=True)
    logger = logging.getLogger(f"test{args.checkpoint}_{args.num_times}")
    logger.info(str(args))

    test_data = load_data_volume(
        data=args.data,
        batch_size=1,
        path_prefix=args.data_dir,
        augmentation=False,
        split="test",
        rand_crop_spatial_size=args.rand_crop_size,
        convert_to_sam=False,
        do_test_crop=False,
        deterministic=True,
        num_worker=4
    )

    sam = sam_model_registry["vit_b"](checkpoint="ckpts/sam_vit_b_01ec64.pth")

    img_encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
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
        embedding_size=32,
        new_window_size=args.new_window_size,
    )
    img_encoder.load_state_dict(sam.image_encoder.state_dict(), strict=False)
    img_encoder.to(device)
    img_encoder.update_pos_embed()

    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(32, 32),
        input_image_size=(512, 512),
        mask_in_chans=16,
    )
    prompt_encoder.load_state_dict(sam.prompt_encoder.state_dict())
    prompt_encoder.to(device)

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
    mask_decoder.load_state_dict(torch.load(os.path.join(save_path, file), map_location='cpu')["decoder_dict"], strict=True)
    mask_decoder.to(device)

    del sam

    img_encoder.eval()
    prompt_encoder.eval()
    mask_decoder.eval()
    dice_loss = DiceLoss(squared_pred=True, reduction="mean")

    patch_size = args.rand_crop_size[0]
    sparse_embeddings_n, dense_embeddings = prompt_encoder(
        points=None,
        boxes=None,
        masks=None,
    )
    image_pe = prompt_encoder.get_dense_pe()

    for n in range(args.num_times):
        output_path = os.path.join(save_path, "_".join([args.data, str.split(str.split(logger.handlers[0].baseFilename, "/")[-1], ".")[0], str(n)]))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with torch.no_grad():
            loss_summary = []
            loss_nsd = []
            for idx, (img, seg, spacing, path, affine) in enumerate(test_data):
                print('seg: ', seg.sum(), path, img.shape, seg.shape)
                seg = seg.unsqueeze(0)
                l = len(np.where(seg == 1)[0])

                if l > 0:
                    sample = np.random.choice(np.arange(l), args.num_points, replace=False)

                    points_x = np.where(seg == 1)[-1][sample]
                    points_y = np.where(seg == 1)[-2][sample]
                    points_z = np.where(seg == 1)[-3][sample]

                logger.info((points_x, points_y, points_z))

                x = points_x * img.shape[-1] / seg.shape[-1]
                y = points_y * img.shape[-2] / seg.shape[-2]
                z = points_z * img.shape[-3] / seg.shape[-3]
                points_crood=(x, y, z)

                pred = sliding_window_inference(img, [patch_size, patch_size, patch_size], overlap=0.5, sw_batch_size=1,
                                                mode="gaussian",
                                                sigma_scale=0.25,
                                                device=torch.device("cpu"),
                                                predictor=partial(model_predict,
                                                                  points_crood=points_crood,
                                                                  img_encoder=img_encoder,
                                                                  prompt_encoder=prompt_encoder,
                                                                  mask_decoder=mask_decoder,
                                                                  sparse_embeddings_n=sparse_embeddings_n,
                                                                  dense_embeddings=dense_embeddings,
                                                                  image_pe=image_pe,
                                                                  ),
                                                with_coord=True)

                final_pred = F.interpolate(pred, size=seg.shape[2:], mode="trilinear")
                final_pred = F.sigmoid(final_pred)
                masks = final_pred > 0.5
                loss = 1 - dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())

                ssd = SD.compute_surface_distances((seg == 1)[0, 0].cpu().numpy(), (masks==1)[0, 0].cpu().numpy(), spacing_mm=spacing[0].numpy())
                nsd = SD.compute_surface_dice_at_tolerance(ssd, args.tolerance)
                loss_nsd.append(nsd)
                logger.info(" Case {} - Dice {:.6f} | NSD {:.6f}".format(test_data.dataset.img_dict[idx], loss.item(), nsd))

                if args.save_masks:
                    name = path[0].split("/")[-2] + path[0].split("/")[-1]
                    writer = NibabelWriter()
                    image_data_pre = np.uint8(masks.squeeze(0).squeeze(0).permute(2, 1, 0).detach().cpu())
                    affine = affine.squeeze(0).detach().cpu().numpy()
                    writer.set_data_array(image_data_pre, channel_dim=None)
                    writer.set_metadata({"affine": affine})
                    writer.write(f"{output_path}/{args.data}_pred_{name}", verbose=True)

            logger.info("- Test metrics Dice: " + str(np.mean(loss_summary)))
            logger.info("- Test metrics NSD: " + str(np.mean(loss_nsd)))


if __name__ == "__main__":
    main()

