from functools import partial

import torch
import torch.nn as nn
from util.pos_embed import get_1d_sincos_pos_embed
from sc_patchemb import PatchEmbed
from util.my_flashattn import FlashAttentionBlock
from timm.models.vision_transformer import Block, Attention
import torch.nn.functional as F


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, gene_size=20400, patch_size=100, in_chans=3,
                 embed_dim=16, depth=24, num_heads=16,
                 decoder_embed_dim=16, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 protein_embed_dim=768, similarity_loss_weight=0.1):

        super().__init__()  # <--- 1. 在这里添加这一行

        self.patch_embed = PatchEmbed(gene_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, attn_drop=0.1, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size)
        self.protein_embed_dim = protein_embed_dim  #
        self.similarity_loss_weight = similarity_loss_weight  #
        self.similarity_loss_sample_size = 32
        # --------------------------------------------------------------------------
        self.activation = nn.ReLU()
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches),
                                                    cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, cells):
        p = self.patch_embed.patch_size
        assert cells.shape[1] % p == 0

        n = cells.shape[1] // p  # 16900/100
        x = cells.reshape(shape=(cells.shape[0], n, -1))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size
        x = x.reshape(shape=(x.shape[0], -1))
        return x

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))  #

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  #
        ids_keep = ids_shuffle[:, :len_keep]  #
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  #
        mask = torch.ones([N, L], device=x.device)  #
        mask[:, :len_keep] = 0  #
        mask = torch.gather(mask, dim=1, index=ids_restore)  #
        return x_masked, mask, ids_restore, ids_keep  # <--- 修改

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)  #
        x = x + self.pos_embed[:, 1:, :]  #
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)  # <--- 修改
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  #
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  #
        x = torch.cat((cls_tokens, x), dim=1)  #
        x = torch.nan_to_num(x)  #
        for blk in self.blocks:  #
            x = blk(x)  # (N, L, D) #
        x = self.norm(x)  #
        return x, mask, ids_restore, ids_keep  # <--- 修改

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = self.activation(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, cells, pred, mask):
        target = self.patchify(cells)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, cells, protein_targets, mask_ratio=0.75):

        latent, mask, ids_restore, ids_keep = self.forward_encoder(cells, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss_recon = self.forward_loss(cells, pred, mask)
        B_patches_all = latent[:, 1:, :]  # Shape: [N, len_keep, D_latent]
        P_patches_all = torch.gather(
            protein_targets,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, self.protein_embed_dim)
        )  # Shape: [N, len_keep, D_protein]

        N, L_keep, _ = B_patches_all.shape

        if N > self.similarity_loss_sample_size:
            sample_idx = torch.randperm(N, device=latent.device)[:self.similarity_loss_sample_size]
            B_patches_final = B_patches_all[sample_idx]
            P_patches_final = P_patches_all[sample_idx]
        else:
            B_patches_final = B_patches_all
            P_patches_final = P_patches_all

        B = B_patches_final.reshape(-1, B_patches_final.shape[-1])
        P = P_patches_final.reshape(-1, P_patches_final.shape[-1])
        G = B.shape[0]
        if G == 0:
            loss_sim = torch.tensor(0.0, device=B.device, requires_grad=True)
        else:
            idx_shuffle = torch.randperm(G, device=B.device)
            B_shuffled = B[idx_shuffle]  # B: GeneByte
            P_shuffled = P[idx_shuffle]  # P: ProteinEmbed

            dist_B = 1.0 - F.cosine_similarity(B, B_shuffled, dim=1)
            dist_P = 1.0 - F.cosine_similarity(P, P_shuffled, dim=1)

            loss_sim = F.mse_loss(dist_B, dist_P)

        loss = loss_recon + self.similarity_loss_weight * loss_sim

        return {"loss": loss, "loss_recon": loss_recon, "loss_sim": loss_sim}


def mae_vit_base_patch16_dim256(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=256, depth=12, num_heads=8,
        decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_patch16_dim256 = mae_vit_base_patch16_dim256

