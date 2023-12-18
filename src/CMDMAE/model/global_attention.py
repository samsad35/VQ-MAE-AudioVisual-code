import torch
from einops import repeat, rearrange
from timm.models.layers import trunc_normal_
from timm.models.layers import Mlp


class CrossAttention(torch.nn.Module):
    def __init__(self,
                 embed_dim=None,
                 num_heads=None,
                 mlp_ratio=2.0):
        super(CrossAttention, self).__init__()
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(mlp_ratio * embed_dim))

    def forward(self, query, key, value, return_attention=False):
        attn_output, att = self.cross_attention(query, key, value)
        attn_output = attn_output + query
        if return_attention:
            return attn_output + self.layer_norm(self.mlp(attn_output)), att
        else:
            return attn_output + self.layer_norm(self.mlp(attn_output))


class AttentionPooling(torch.nn.Module):
    def __init__(self, latent_dim: int, num_heads: int, mlp_ratio: float):
        super(AttentionPooling, self).__init__()
        self.cls_token = torch.nn.Parameter(torch.randn(1, latent_dim))
        self.cross_attention = CrossAttention(embed_dim=latent_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, input, return_attention=False):  # input (T, batch, dim=latent_dim)
        query = repeat(self.cls_token, 'n d ->  n b d', b=input.shape[1])
        key, value = input, input
        return self.cross_attention(query, key, value, return_attention=return_attention)


if __name__ == '__main__':
    attention_pooling = AttentionPooling(latent_dim=512, num_heads=4, mlp_ratio=2)
    x = torch.randn((50, 25, 512))
    cls = attention_pooling(x)
    print(cls.shape)
