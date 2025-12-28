import numpy as np

import torch



def get_1d_sincos_pos_embed(embed_dim, n_pos, cls_token):
    """
    embed_dim: output dimension for each position
    n_pos: dimension of patch
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    emb = np.zeros([n_pos, embed_dim])

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    #pos = pos.reshape(-1)  # (M,)
    pos = np.arange(n_pos, dtype=np.float64)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    #emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    emb[:, 0::2] = emb_sin
    emb[:, 1::2] = emb_cos
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    else:
        pos_embed = emb

    return pos_embed


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = pos_embed_checkpoint.shape[-2] - num_extra_tokens
        # height (== width) for the new position embedding
        new_size = num_patches
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %d to %d" % (orig_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, embedding_size).permute(0, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=new_size)
            pos_tokens = pos_tokens.permute(0, 2, 1)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
