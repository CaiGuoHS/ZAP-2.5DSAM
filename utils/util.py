'''
From https://github.com/med-air/3DSAM-adapter/blob/main/3DSAM-adapter/utils/util.py
'''
import shutil
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

def save_checkpoint(state, is_best, checkpoint):
    filepath_last = os.path.join(checkpoint, "last.pth.tar")
    filepath_best = os.path.join(checkpoint, "best.pth.tar")
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Masking directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint DIrectory exists!")
    torch.save(state, filepath_last)
    if is_best:
        if os.path.isfile(filepath_best):
            os.remove(filepath_best)
        shutil.copyfile(filepath_last, filepath_best)


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(logger_name, get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime


def model_predict(img, crood, points_crood, img_encoder, prompt_encoder, mask_decoder, sparse_embeddings_n, dense_embeddings, image_pe, patch_size=128, empty_point=True, device="cuda:0"):
    img = img.to(device)
    crood_d_min = crood[0][-3].start
    crood_h_min = crood[0][-2].start
    crood_w_min = crood[0][-1].start
    points_crood_x = points_crood[0]
    points_crood_y = points_crood[1]
    points_crood_z = points_crood[2]

    crood_x = points_crood_x - crood_w_min
    crood_y = points_crood_y - crood_h_min
    crood_z = points_crood_z - crood_d_min

    is_empty = True
    points_x = []
    points_y = []
    points_z = []
    for i in range(len(crood_x)):
        if (0 <= crood_x[i] <= patch_size-1) and (0 <= crood_y[i] <= patch_size-1) and (0 <= crood_z[i] <= patch_size-1):
            is_empty = False
            points_x.append(crood_x[i])
            points_y.append(crood_y[i])
            points_z.append(crood_z[i])

    if is_empty and empty_point:
        pred = torch.zeros_like(img[:, :1]).to(img.device)
    else:
        with torch.no_grad():
            input_batch_d = F.interpolate(img[0].permute(1, 0, 2, 3), scale_factor=512/patch_size, mode='bilinear')
            image_embeddings_d = img_encoder(input_batch_d)
            input_batch_h = F.interpolate(img[0].permute(2, 0, 3, 1), scale_factor=512/patch_size, mode='bilinear')
            image_embeddings_h = img_encoder(input_batch_h)
            input_batch_w = F.interpolate(img[0].permute(3, 0, 1, 2), scale_factor=512/patch_size, mode='bilinear')
            image_embeddings_w = img_encoder(input_batch_w)

            if is_empty:
                sparse_embeddings_d = sparse_embeddings_n
                sparse_embeddings_h = sparse_embeddings_n
                sparse_embeddings_w = sparse_embeddings_n
            else:
                num_samples = len(points_x)
                x = np.array(points_x) * 512 / patch_size
                y = np.array(points_y) * 512 / patch_size
                z = np.array(points_z) * 512 / patch_size
                points_d = [[x[i], y[i]] for i in range(num_samples)]
                points_h = [[z[i], x[i]] for i in range(num_samples)]
                points_w = [[y[i], z[i]] for i in range(num_samples)]
                points_d_torch = torch.tensor(points_d, dtype=torch.float32, device=img.device).unsqueeze(0)
                points_w_torch = torch.tensor(points_w, dtype=torch.float32, device=img.device).unsqueeze(0)
                points_h_torch = torch.tensor(points_h, dtype=torch.float32, device=img.device).unsqueeze(0)
                labels = [1 for _ in range(num_samples)]
                labels_torch = torch.tensor(labels, dtype=torch.int32, device=img.device).unsqueeze(0)

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

            pred = mask_decoder(
                image_embeddings=(image_embeddings_d, image_embeddings_h, image_embeddings_w),
                image_pe=image_pe,
                sparse_prompt_embeddings=(sparse_embeddings_d, sparse_embeddings_h, sparse_embeddings_w),
                dense_prompt_embeddings=dense_embeddings,
            )
    return pred.cpu()


def model_predict_time(img, crood, points_crood, img_encoder, prompt_encoder, mask_decoder, sparse_embeddings_n, dense_embeddings, image_pe, logger_time, patch_size=128, empty_point=True, device="cuda:0"):
    img = img.to(device)
    crood_d_min = crood[0][-3].start
    crood_h_min = crood[0][-2].start
    crood_w_min = crood[0][-1].start
    points_crood_x = points_crood[0]
    points_crood_y = points_crood[1]
    points_crood_z = points_crood[2]

    crood_x = points_crood_x - crood_w_min
    crood_y = points_crood_y - crood_h_min
    crood_z = points_crood_z - crood_d_min

    is_empty = True
    points_x = []
    points_y = []
    points_z = []
    for i in range(len(crood_x)):
        if (0 <= crood_x[i] <= patch_size-1) and (0 <= crood_y[i] <= patch_size-1) and (0 <= crood_z[i] <= patch_size-1):
            is_empty = False
            points_x.append(crood_x[i])
            points_y.append(crood_y[i])
            points_z.append(crood_z[i])

    if is_empty and empty_point:
        pred = torch.zeros_like(img[:, :1]).to(img.device)
    else:
        torch.cuda.synchronize()
        time_start = time.time()
        with torch.no_grad():
            input_batch_d = F.interpolate(img[0].permute(1, 0, 2, 3), scale_factor=512/patch_size, mode='bilinear')
            image_embeddings_d = img_encoder(input_batch_d)
            input_batch_h = F.interpolate(img[0].permute(2, 0, 3, 1), scale_factor=512/patch_size, mode='bilinear')
            image_embeddings_h = img_encoder(input_batch_h)
            input_batch_w = F.interpolate(img[0].permute(3, 0, 1, 2), scale_factor=512/patch_size, mode='bilinear')
            image_embeddings_w = img_encoder(input_batch_w)

            if is_empty:
                sparse_embeddings_d = sparse_embeddings_n
                sparse_embeddings_h = sparse_embeddings_n
                sparse_embeddings_w = sparse_embeddings_n
            else:
                num_samples = len(points_x)
                x = np.array(points_x) * 512 / patch_size
                y = np.array(points_y) * 512 / patch_size
                z = np.array(points_z) * 512 / patch_size
                points_d = [[x[i], y[i]] for i in range(num_samples)]
                points_h = [[z[i], x[i]] for i in range(num_samples)]
                points_w = [[y[i], z[i]] for i in range(num_samples)]
                points_d_torch = torch.tensor(points_d, dtype=torch.float32, device=img.device).unsqueeze(0)
                points_w_torch = torch.tensor(points_w, dtype=torch.float32, device=img.device).unsqueeze(0)
                points_h_torch = torch.tensor(points_h, dtype=torch.float32, device=img.device).unsqueeze(0)
                labels = [1 for _ in range(num_samples)]
                labels_torch = torch.tensor(labels, dtype=torch.int32, device=img.device).unsqueeze(0)

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

            pred = mask_decoder(
                image_embeddings=(image_embeddings_d, image_embeddings_h, image_embeddings_w),
                image_pe=image_pe,
                sparse_prompt_embeddings=(sparse_embeddings_d, sparse_embeddings_h, sparse_embeddings_w),
                dense_prompt_embeddings=dense_embeddings,
            )
        torch.cuda.synchronize()
        time_stop = time.time()
        logger_time.info(f"{(time_stop-time_start):.4f}")
    return pred.cpu()