# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import math

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from typing import Tuple, Type, List

from segment_anything.modeling.common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # self.output_upscaling = nn.Sequential(
        #     nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1),
        #     LayerNorm2d(transformer_dim // 4),
        #     activation(),
        #     # nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=1),
        #     # activation(),
        # )
        #
        # self.output_upscaling2 = nn.Sequential(
        #     # nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1),
        #     # LayerNorm2d(transformer_dim // 4),
        #     # activation(),
        #     nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=1),
        #     activation(),
        # )

        self.output_hypernetworks_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        # self.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #         for i in range(4)
        #     ]
        # )

    def forward(
        self,
        image_embeddings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dense_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:

        d, c, h, w = image_embeddings[0].shape

        pos_src = image_pe
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings[0].size(0), -1, -1)

        tokens_d = torch.cat((output_tokens, sparse_prompt_embeddings[0]), dim=1)
        src_d = image_embeddings[0]
        src_d = src_d + dense_prompt_embeddings
        hs_d, src_d = self.transformer(src_d, pos_src, tokens_d)
        mask_tokens_out_d = hs_d[:, 1:2, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src_d = src_d.transpose(1, 2).reshape(d, c, h, w)
        upscaled_embedding_d = self.output_upscaling(src_d)
        hyper_in_d = self.output_hypernetworks_mlp(mask_tokens_out_d[:, 0, :]).unsqueeze(1)
        D, C, H, W = upscaled_embedding_d.shape
        masks_d = (hyper_in_d @ upscaled_embedding_d.view(D, C, H * W)).view(D, -1, H, W).unsqueeze(0).permute(0, 2, 1, 3, 4)
        masks_d = F.interpolate(masks_d, scale_factor=(4, 1, 1), mode="trilinear")


        tokens_h = torch.cat((output_tokens, sparse_prompt_embeddings[1]), dim=1)
        src_h = image_embeddings[1]
        src_h = src_h + dense_prompt_embeddings
        hs_h, src_h = self.transformer(src_h, pos_src, tokens_h)
        mask_tokens_out_h = hs_h[:, 1:2, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src_h = src_h.transpose(1, 2).reshape(h, c, w, d)
        upscaled_embedding_h = self.output_upscaling(src_h)
        hyper_in_h = self.output_hypernetworks_mlp(mask_tokens_out_h[:, 0, :]).unsqueeze(1)
        H, C, W, D = upscaled_embedding_h.shape
        masks_h = (hyper_in_h @ upscaled_embedding_h.view(H, C, W * D)).view(H, -1, W, D).unsqueeze(0).permute(0, 2, 4, 1, 3)
        masks_h = F.interpolate(masks_h, scale_factor=(1, 4, 1), mode="trilinear")


        tokens_w = torch.cat((output_tokens, sparse_prompt_embeddings[2]), dim=1)
        src_w = image_embeddings[2]
        src_w = src_w + dense_prompt_embeddings
        hs_w, src_w = self.transformer(src_w, pos_src, tokens_w)
        mask_tokens_out_w = hs_w[:, 1:2, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src_w = src_w.transpose(1, 2).reshape(w, c, d, h)
        upscaled_embedding_w = self.output_upscaling(src_w)
        hyper_in_w = self.output_hypernetworks_mlp(mask_tokens_out_w[:, 0, :]).unsqueeze(1)
        W, C, D, H = upscaled_embedding_w.shape
        masks_w = (hyper_in_w @ upscaled_embedding_w.view(W, C, D * H)).view(W, -1, D, H).unsqueeze(0).permute(0, 2, 3, 4, 1)
        masks_w = F.interpolate(masks_w, scale_factor=(1, 1, 4), mode="trilinear")

        final_masks = masks_d + masks_h + masks_w

        return final_masks


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
