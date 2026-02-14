
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous()  # 2, B, num_heads, prompt_length, C // num_heads
            key_prefix = prompt[0]  # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1]  # B, num_heads, prompt_length, embed_dim // num_heads

            # The original code had assertions here checking shapes against k and v.
            # B, num_heads, prompt_length, embed_dim // num_heads
            # vs
            # B, num_heads, N, embed_dim // num_heads
            # The prompt length is different from N (sequence length), so we can't assert strict equality on dim 2.
            # However, the original code might have been checking batch/heads/dim consistency.
            # The original code:
            # expected_shape = (B, self.num_heads, C // self.num_heads)
            # assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape
            
            # We will keep the logic as is from the source I read.
            expected_shape = (B, self.num_heads, C // self.num_heads)
            # Note: stored prompt shape is [B, num_heads, prompt_len, head_dim]
            # key_prefix.shape[0] is B
            # key_prefix.shape[1] is num_heads
            # key_prefix.shape[3] is head_dim (C // num_heads)
            
            # So the checks are valid.
            
            # k shape is [B, num_heads, N, head_dim]
            
            # Logic check:
            # assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape
            
            # Porting exactly as read.

            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
