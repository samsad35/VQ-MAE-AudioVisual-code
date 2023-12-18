import torch
from torch.distributions.dirichlet import Dirichlet
import numpy as np
from einops import repeat, rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from timm.models.layers import Mlp
import math
from .global_attention import AttentionPooling

"""
    Others
"""


def random_indexes(size: int, random: bool = True):
    forward_indexes = np.arange(size)
    if random:
        np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class TokenShuffle(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tokens: torch.Tensor, ratio, random: bool = True):
        T, B, C = tokens.shape
        remain_T = torch.round(T * (1 - ratio)).to(torch.int)
        indexes = [random_indexes(T, random=random) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(
            tokens.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(
            tokens.device)
        tokens = take_indexes(tokens, forward_indexes)
        tokens = tokens[:remain_T]
        return tokens, forward_indexes, backward_indexes


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return self.pe[:x.size(0)]


"""
    Encoder CMDMAE
"""


class Encoder(torch.nn.Module):
    """
        Encoder Of Contrastive Multimodal Dynamical Masked AutoEncoder
    """

    def __init__(self,
                 seq_length_v: int = None,
                 seq_length_a: int = None,
                 num_embeddings_v: int = None,
                 emb_dim_v: int = None,
                 num_indices_v: int = None,
                 num_indices_a: int = None,
                 num_embeddings_a: int = None,
                 emb_dim_a: int = None,
                 num_layer: int = None,
                 num_head: int = None,
                 alpha: tuple = None,
                 pos_embedding_trained: bool = True,
                 vqvae_v_embedding=None,
                 vqvae_a_embedding=None,
                 mlp_ratio=4.0
                 ):
        super(Encoder, self).__init__()
        self.dirichlet = Dirichlet(torch.tensor([alpha[0], alpha[1]]))
        self.proj_v = torch.nn.Embedding(num_embeddings=num_embeddings_v, embedding_dim=emb_dim_v)
        self.proj_a = torch.nn.Embedding(num_embeddings=num_embeddings_a, embedding_dim=emb_dim_a)
        self.modality_emb_v = torch.nn.Parameter(torch.zeros(1, 1, emb_dim_v * num_indices_v))
        self.modality_emb_a = torch.nn.Parameter(torch.zeros(1, 1, emb_dim_a * num_indices_a))
        if pos_embedding_trained:
            self.pos_embedding_v = torch.nn.Parameter(torch.zeros(seq_length_v, 1, num_indices_v * emb_dim_v))
            self.pos_embedding_a = torch.nn.Parameter(torch.zeros(seq_length_a, 1, num_indices_a * emb_dim_a))
        else:
            self.pos_embedding_v = PositionalEncoding(d_model=num_indices_v * emb_dim_v, max_len=seq_length_v)
            self.pos_embedding_a = PositionalEncoding(d_model=num_indices_a * emb_dim_a, max_len=seq_length_a)
        self.transformer = \
            torch.nn.Sequential(
                *[Block(num_indices_v * emb_dim_v, num_head, mlp_ratio=mlp_ratio) for _ in range(num_layer)])
        self.shuffle = TokenShuffle()
        self.layer_norm = torch.nn.LayerNorm(num_indices_v * emb_dim_v)
        self.pos_embedding_trained = pos_embedding_trained
        self.vqvae_v_embedding = vqvae_v_embedding
        self.vqvae_a_embedding = vqvae_a_embedding
        self.init_weight()
        """ Parameters """
        self.emb_dim_v = emb_dim_v
        self.num_indices_v = num_indices_v
        self.emb_dim_a = emb_dim_a
        self.num_indices_a = num_indices_a
        self.seq_length_v = seq_length_v
        self.seq_length_a = seq_length_a

    def init_weight(self):
        if self.pos_embedding_trained:
            trunc_normal_(self.pos_embedding_v, std=.02)
            trunc_normal_(self.pos_embedding_a, std=.02)
        trunc_normal_(self.modality_emb_v, std=.02)
        trunc_normal_(self.modality_emb_a, std=.02)
        if self.vqvae_v_embedding is not None:
            self.proj_v = torch.nn.Embedding.from_pretrained(embeddings=self.vqvae_v_embedding, freeze=True)
        if self.vqvae_a_embedding is not None:
            self.proj_a = torch.nn.Embedding.from_pretrained(embeddings=self.vqvae_a_embedding, freeze=True)

    def forward(self, x_v, x_a, ratio=None):
        # Visual tokens + pos_embedding + modality embedding
        x_v = rearrange(x_v, 'b t c -> t b c')
        x_v = self.proj_v(x_v).reshape(self.seq_length_v, -1, self.num_indices_v * self.emb_dim_v)
        if self.pos_embedding_trained:
            x_v = x_v + self.pos_embedding_v + self.modality_emb_v
        else:
            x_v = x_v + self.pos_embedding_v(x_v) + self.modality_emb_v

        # Audio tokens + pos_embedding + modality embedding
        x_a = rearrange(x_a, 'b t c -> t b c')
        x_a = self.proj_a(x_a).reshape(self.seq_length_a, -1, self.num_indices_a * self.emb_dim_a)
        if self.pos_embedding_trained:
            x_a = x_a + self.pos_embedding_a + self.modality_emb_a
        else:
            x_a = x_a + self.pos_embedding_a(x_a) + self.modality_emb_a

        # Shuffle + Transfomer
        if ratio is not None:
            ratio_v, ratio_a = ratio[0], ratio[1]
        else:
            ratio_v, ratio_a = self.dirichlet.sample()
        x_v, forward_indexes_v, backward_indexes_v = self.shuffle(x_v, ratio=ratio_v, random=True)
        x_a, forward_indexes_a, backward_indexes_a = self.shuffle(x_a, ratio=ratio_a, random=True)
        x_av = torch.cat((x_v, x_a), dim=0)
        x_av = rearrange(x_av, 't b c -> b t c')
        z_av = self.layer_norm(self.transformer(x_av))
        z_av = rearrange(z_av, 'b t c -> t b c')
        z_v = z_av[:x_v.shape[0]]
        z_a = z_av[x_v.shape[0]:]
        return z_v, forward_indexes_v, backward_indexes_v, z_a, forward_indexes_a, backward_indexes_a


""" Encoder With Cross-Attention"""


class Encoder_WithCrossAttention(torch.nn.Module):
    """
        Encoder Of Contrastive Multimodal Dynamical Masked AutoEncoder
    """

    def __init__(self,
                 seq_length_v: int = None,
                 seq_length_a: int = None,
                 num_embeddings_v: int = None,
                 emb_dim_v: int = None,
                 num_indices_v: int = None,
                 num_indices_a: int = None,
                 num_embeddings_a: int = None,
                 emb_dim_a: int = None,
                 num_layer: int = None,
                 num_head: int = None,
                 alpha: tuple = None,
                 pos_embedding_trained: bool = True,
                 vqvae_v_embedding=None,
                 vqvae_a_embedding=None,
                 mlp_ratio=4.0
                 ):
        super(Encoder_WithCrossAttention, self).__init__()
        self.dirichlet = Dirichlet(torch.tensor([alpha[0], alpha[1]]))
        self.proj_v = torch.nn.Embedding(num_embeddings=num_embeddings_v, embedding_dim=emb_dim_v)
        self.proj_a = torch.nn.Embedding(num_embeddings=num_embeddings_a, embedding_dim=emb_dim_a)
        self.modality_emb_v = torch.nn.Parameter(torch.zeros(1, 1, emb_dim_v * num_indices_v))
        self.modality_emb_a = torch.nn.Parameter(torch.zeros(1, 1, emb_dim_a * num_indices_a))
        if pos_embedding_trained:
            self.pos_embedding_v = torch.nn.Parameter(torch.zeros(seq_length_v, 1, num_indices_v * emb_dim_v))
            self.pos_embedding_a = torch.nn.Parameter(torch.zeros(seq_length_a, 1, num_indices_a * emb_dim_a))
        else:
            self.pos_embedding_v = PositionalEncoding(d_model=num_indices_v * emb_dim_v, max_len=seq_length_v)
            self.pos_embedding_a = PositionalEncoding(d_model=num_indices_a * emb_dim_a, max_len=seq_length_a)
        self.transformer_v = \
            torch.nn.Sequential(
                *[Block(emb_dim_v * num_indices_v, num_head, mlp_ratio=mlp_ratio) for _ in range(num_layer)])
        self.transformer_a = \
            torch.nn.Sequential(
                *[Block(emb_dim_a * num_indices_a, num_head, mlp_ratio=mlp_ratio) for _ in range(num_layer)])
        self.cross_attention_v = CrossAttention(embed_dim=emb_dim_v * num_indices_v, num_heads=num_head,
                                                mlp_ratio=mlp_ratio)
        self.cross_attention_a = CrossAttention(embed_dim=emb_dim_a * num_indices_a, num_heads=num_head,
                                                mlp_ratio=mlp_ratio)
        self.shuffle = TokenShuffle()
        self.layer_norm_v = torch.nn.LayerNorm(num_indices_v * emb_dim_v)
        self.layer_norm_a = torch.nn.LayerNorm(num_indices_a * emb_dim_a)
        self.pos_embedding_trained = pos_embedding_trained
        self.vqvae_v_embedding = vqvae_v_embedding
        self.vqvae_a_embedding = vqvae_a_embedding
        self.init_weight()
        """ Parameters """
        self.emb_dim_v = emb_dim_v
        self.num_indices_v = num_indices_v
        self.emb_dim_a = emb_dim_a
        self.num_indices_a = num_indices_a
        self.seq_length_v = seq_length_v
        self.seq_length_a = seq_length_a

    def init_weight(self):
        if self.pos_embedding_trained:
            trunc_normal_(self.pos_embedding_v, std=.02)
            trunc_normal_(self.pos_embedding_a, std=.02)
        trunc_normal_(self.modality_emb_v, std=.02)
        trunc_normal_(self.modality_emb_a, std=.02)
        if self.vqvae_v_embedding is not None:
            self.proj_v = torch.nn.Embedding.from_pretrained(embeddings=self.vqvae_v_embedding, freeze=True)
        if self.vqvae_a_embedding is not None:
            self.proj_a = torch.nn.Embedding.from_pretrained(embeddings=self.vqvae_a_embedding, freeze=True)

    def forward(self, x_v, x_a, ratio=None):
        # Visual tokens + pos_embedding + modality embedding
        x_v = rearrange(x_v, 'b t c -> t b c')
        x_v = self.proj_v(x_v).reshape(self.seq_length_v, -1, self.num_indices_v * self.emb_dim_v)
        if self.pos_embedding_trained:
            x_v = x_v + self.pos_embedding_v + self.modality_emb_v
        else:
            x_v = x_v + self.pos_embedding_v(x_v) + self.modality_emb_v

        # Audio tokens + pos_embedding + modality embedding
        x_a = rearrange(x_a, 'b t c -> t b c')
        x_a = self.proj_a(x_a).reshape(self.seq_length_a, -1, self.num_indices_a * self.emb_dim_a)
        if self.pos_embedding_trained:
            x_a = x_a + self.pos_embedding_a + self.modality_emb_a
        else:
            x_a = x_a + self.pos_embedding_a(x_a) + self.modality_emb_a

        # Shuffle + Transfomer
        if ratio is not None:
            ratio_v, ratio_a = ratio[0], ratio[1]
        else:
            ratio_v, ratio_a = self.dirichlet.sample()
        x_v, forward_indexes_v, backward_indexes_v = self.shuffle(x_v, ratio=ratio_v)
        x_a, forward_indexes_a, backward_indexes_a = self.shuffle(x_a, ratio=ratio_a)

        x_av = torch.cat((x_a, x_v), dim=0)

        # audio
        x_a = self.cross_attention_a(x_a, x_av, x_av)  # Query=x_a
        x_a = rearrange(x_a, 't b c -> b t c')
        z_a = self.layer_norm_a(self.transformer_a(x_a))
        z_a = rearrange(z_a, 'b t c -> t b c')

        # visual
        x_v = self.cross_attention_v(x_v, x_av, x_av)  # Query=x_v
        x_v = rearrange(x_v, 't b c -> b t c')
        z_v = self.layer_norm_v(self.transformer_v(x_v))
        z_v = rearrange(z_v, 'b t c -> t b c')

        return z_v, forward_indexes_v, backward_indexes_v, z_a, forward_indexes_a, backward_indexes_a


"""
    Decoder CMDMAE
"""

""" Decoder without Cross-Attention"""


class Decoder_WithoutCrossAttention(torch.nn.Module):
    """
        Decoder Without Cross-Attention
    """

    def __init__(self,
                 seq_length_v: int = None,
                 seq_length_a: int = None,
                 emb_dim_v: int = None,
                 num_indices_v: int = None,
                 num_indices_a: int = None,
                 emb_dim_a: int = None,
                 num_layer: int = None,
                 num_head: int = None,
                 dim_tokens_v: int = None,
                 dim_tokens_a: int = None,
                 pos_embedding_trained: bool = True,
                 mlp_ratio=4.0
                 ):
        super(Decoder_WithoutCrossAttention, self).__init__()
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim_v * num_indices_v))
        self.modality_emb_v = torch.nn.Parameter(torch.zeros(1, 1, emb_dim_v * num_indices_v))
        self.modality_emb_a = torch.nn.Parameter(torch.zeros(1, 1, emb_dim_a * num_indices_a))
        if pos_embedding_trained:
            self.pos_embedding_v = torch.nn.Parameter(torch.zeros(seq_length_v, 1, num_indices_v * emb_dim_v))
            self.pos_embedding_a = torch.nn.Parameter(torch.zeros(seq_length_a, 1, num_indices_a * emb_dim_a))
        else:
            self.pos_embedding_v = PositionalEncoding(d_model=num_indices_v * emb_dim_v, max_len=seq_length_v)
            self.pos_embedding_a = PositionalEncoding(d_model=num_indices_a * emb_dim_a, max_len=seq_length_a)
        self.transformer = \
            torch.nn.Sequential(
                *[Block(emb_dim_v * num_indices_v, num_head, mlp_ratio=mlp_ratio) for _ in range(num_layer)])
        self.head_v = torch.nn.Linear(emb_dim_v, dim_tokens_v)
        self.head_a = torch.nn.Linear(emb_dim_a, dim_tokens_a)
        self.pos_embedding_trained = pos_embedding_trained
        self.init_weight()

        """ Parameters """
        self.seq_length_v = seq_length_v
        self.seq_length_a = seq_length_a
        self.num_indices_v = num_indices_v
        self.num_indices_a = num_indices_a

    def init_weight(self):
        if self.pos_embedding_trained:
            trunc_normal_(self.pos_embedding_v, std=.02)
            trunc_normal_(self.pos_embedding_a, std=.02)
        trunc_normal_(self.modality_emb_v, std=.02)
        trunc_normal_(self.modality_emb_a, std=.02)

    def forward(self, z_v, backward_indexes_v, z_a, backward_indexes_a):
        T_v, T_a = z_v.shape[0], z_a.shape[0]
        # Visual tokens + pos_embedding + modality embedding
        z_v = torch.cat([z_v, self.mask_token.expand(backward_indexes_v.shape[0] - z_v.shape[0], z_v.shape[1], -1)],
                        dim=0)
        z_v = take_indexes(z_v, backward_indexes_v)
        if self.pos_embedding_trained:
            z_v = z_v + self.pos_embedding_v + self.modality_emb_v
        else:
            z_v = z_v + self.pos_embedding_v(z_v) + self.modality_emb_v

        # Audio tokens + pos_embedding + modality embedding
        z_a = torch.cat([z_a, self.mask_token.expand(backward_indexes_a.shape[0] - z_a.shape[0], z_a.shape[1], -1)],
                        dim=0)
        z_a = take_indexes(z_a, backward_indexes_a)
        if self.pos_embedding_trained:
            z_a = z_a + self.pos_embedding_a + self.modality_emb_a
        else:
            z_a = z_a + self.pos_embedding_a(z_a) + self.modality_emb_a
        # Transformer
        z_av = torch.cat((z_v, z_a), dim=0)
        z_av = rearrange(z_av, 't b c -> b t c')
        z_av = self.transformer(z_av)
        z_av = rearrange(z_av, 'b t c -> t b c')

        # Mask + Outputs
        mask_v = torch.zeros_like(z_av[:self.seq_length_v])
        mask_v[T_v:] = 1
        mask_v = take_indexes(mask_v, backward_indexes_v)
        mask_v = rearrange(mask_v, 't b (c d) -> b t c d', c=self.num_indices_v)[:, :, :, 0]

        mask_a = torch.zeros_like(z_av[:self.seq_length_a])
        mask_a[T_a:] = 1
        mask_a = take_indexes(mask_a, backward_indexes_a)
        mask_a = rearrange(mask_a, 't b (c d) -> b t c d', c=self.num_indices_a)[:, :, :, 0]

        z_v = z_av[:self.seq_length_v]
        z_v = rearrange(z_v, 't b (c d) -> b t c d', c=self.num_indices_v)
        z_v = self.head_v(z_v)

        z_a = z_av[self.seq_length_v:]
        z_a = rearrange(z_a, 't b (c d) -> b t c d', c=self.num_indices_a)
        z_a = self.head_a(z_a)
        return z_v, mask_v, z_a, mask_a


""" Decoder With Cross-Attention"""


class Decoder_WithCrossAttention(torch.nn.Module):
    """
        Decoder With Cross-Attention
    """

    def __init__(self,
                 seq_length_v=None,
                 seq_length_a=None,
                 emb_dim_v: int = None,
                 num_indices_v: int = None,
                 num_indices_a: int = None,
                 emb_dim_a: int = None,
                 num_layer: int = None,
                 num_head: int = None,
                 dim_tokens_v: int = None,
                 dim_tokens_a: int = None,
                 pos_embedding_trained: bool = True,
                 mlp_ratio=4.0
                 ):
        super(Decoder_WithCrossAttention, self).__init__()
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim_v * num_indices_v))
        self.modality_emb_v = torch.nn.Parameter(torch.zeros(1, 1, emb_dim_v * num_indices_v))
        self.modality_emb_a = torch.nn.Parameter(torch.zeros(1, 1, emb_dim_a * num_indices_a))
        if pos_embedding_trained:
            self.pos_embedding_v = torch.nn.Parameter(torch.zeros(seq_length_v, 1, num_indices_v * emb_dim_v))
            self.pos_embedding_a = torch.nn.Parameter(torch.zeros(seq_length_a, 1, num_indices_a * emb_dim_a))
        else:
            self.pos_embedding_v = PositionalEncoding(d_model=num_indices_v * emb_dim_v, max_len=seq_length_v)
            self.pos_embedding_a = PositionalEncoding(d_model=num_indices_a * emb_dim_a, max_len=seq_length_a)
        self.transformer_v = \
            torch.nn.Sequential(
                *[Block(emb_dim_v * num_indices_v, num_head, mlp_ratio=mlp_ratio) for _ in range(num_layer)])
        self.transformer_a = \
            torch.nn.Sequential(
                *[Block(emb_dim_a * num_indices_a, num_head, mlp_ratio=mlp_ratio) for _ in range(num_layer)])
        self.head_v = torch.nn.Linear(emb_dim_v, dim_tokens_v)
        self.head_a = torch.nn.Linear(emb_dim_a, dim_tokens_a)
        self.cross_attention_v = CrossAttention(embed_dim=emb_dim_v * num_indices_v, num_heads=num_head,
                                                mlp_ratio=mlp_ratio)
        self.cross_attention_a = CrossAttention(embed_dim=emb_dim_a * num_indices_a, num_heads=num_head,
                                                mlp_ratio=mlp_ratio)
        self.pos_embedding_trained = pos_embedding_trained
        self.init_weight()

        """ Parameters """
        self.seq_length_v = seq_length_v
        self.seq_length_a = seq_length_a
        self.num_indices_v = num_indices_v
        self.num_indices_a = num_indices_a

    def init_weight(self):
        if self.pos_embedding_trained:
            trunc_normal_(self.pos_embedding_v, std=.02)
            trunc_normal_(self.pos_embedding_a, std=.02)
        trunc_normal_(self.modality_emb_v, std=.02)
        trunc_normal_(self.modality_emb_a, std=.02)

    def forward(self, z_v, backward_indexes_v, z_a, backward_indexes_a):
        T_v, T_a = z_v.shape[0], z_a.shape[0]
        # Visual tokens + pos_embedding + modality embedding
        z_v = torch.cat([z_v, self.mask_token.expand(backward_indexes_v.shape[0] - z_v.shape[0], z_v.shape[1], -1)],
                        dim=0)
        z_v = take_indexes(z_v, backward_indexes_v)
        if self.pos_embedding_trained:
            z_v = z_v + self.pos_embedding_v + self.modality_emb_v
        else:
            z_v = z_v + self.pos_embedding_v(z_v) + self.modality_emb_v

        # Audio tokens + pos_embedding + modality embedding
        z_a = torch.cat([z_a, self.mask_token.expand(backward_indexes_a.shape[0] - z_a.shape[0], z_a.shape[1], -1)],
                        dim=0)
        z_a = take_indexes(z_a, backward_indexes_a)
        if self.pos_embedding_trained:
            z_a = z_a + self.pos_embedding_a + self.modality_emb_a
        else:
            z_a = z_a + self.pos_embedding_a(z_a) + self.modality_emb_a

        z_av = torch.cat((z_v, z_a), dim=0)
        # Transformer + CrossAttention
        z_v = self.cross_attention_v(z_v, z_av, z_av)  # Query=z_v
        z_v = rearrange(z_v, 't b c -> b t c')
        z_v = self.transformer_v(z_v)
        z_v = rearrange(z_v, 'b t c -> t b c')

        z_a = self.cross_attention_v(z_a, z_av, z_av)  # Query=z_a
        z_a = rearrange(z_a, 't b c -> b t c')
        z_a = self.transformer_a(z_a)
        z_a = rearrange(z_a, 'b t c -> t b c')

        # Mask + Outputs
        mask_v = torch.zeros_like(z_v)
        mask_v[T_v:] = 1
        mask_v = take_indexes(mask_v, backward_indexes_v)
        mask_v = rearrange(mask_v, 't b (c d) -> b t c d', c=self.num_indices_v)[:, :, :, 0]

        mask_a = torch.zeros_like(z_a)
        mask_a[T_a:] = 1
        mask_a = take_indexes(mask_a, backward_indexes_a)
        mask_a = rearrange(mask_a, 't b (c d) -> b t c d', c=self.num_indices_a)[:, :, :, 0]

        z_v = rearrange(z_v, 't b (c d) -> b t c d', c=self.num_indices_v)
        z_v = self.head_v(z_v)

        z_a = rearrange(z_a, 't b (c d) -> b t c d', c=self.num_indices_a)
        z_a = self.head_a(z_a)
        return z_v, mask_v, z_a, mask_a


class CrossAttention(torch.nn.Module):
    def __init__(self,
                 embed_dim=None,
                 num_heads=None,
                 mlp_ratio=2.0):
        super(CrossAttention, self).__init__()
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(mlp_ratio * embed_dim))

    def forward(self, query, key, value):
        attn_output, _ = self.cross_attention(query, key, value)
        return query + self.layer_norm(self.mlp(attn_output))


class CrossAttention_2(torch.nn.Module):
    def __init__(self,
                 embed_dim=None,
                 num_heads=None,
                 mlp_ratio=2.0):
        super(CrossAttention_2, self).__init__()
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(mlp_ratio * embed_dim))

    def forward(self, query, key, value):
        attn_output, _ = self.cross_attention(query, key, value)
        out = query + self.layer_norm_1(attn_output)
        out = out + self.layer_norm_1(self.mlp(out))
        return out


"""
    CMDMAE
"""


class CMDMAE(torch.nn.Module):
    def __init__(self,
                 seq_length_v: int = None,
                 seq_length_a: int = None,
                 num_embeddings_v: int = None,
                 emb_dim_v: int = None,
                 num_indices_v: int = None,
                 num_indices_a: int = None,
                 num_embeddings_a: int = None,
                 emb_dim_a: int = None,
                 encoder_num_layer: int = None,
                 encoder_num_head: int = None,
                 decoder_num_layer: int = None,
                 decoder_num_head: int = None,
                 dim_tokens_v: int = None,
                 dim_tokens_a: int = None,
                 alpha: tuple = None,
                 decoder_cross_attention=False,
                 encoder_cross_attention=False,
                 pos_embedding_trained: bool = True,
                 vqvae_v_embedding=None,
                 vqvae_a_embedding=None,
                 mlp_ratio=4.0,
                 contrastive: bool = False
                 ):
        super(CMDMAE, self).__init__()
        self.decoder_cross_attention = decoder_cross_attention
        self.encoder_cross_attention = encoder_cross_attention
        self.encoder_num_head = encoder_num_head
        self.mlp_ratio = mlp_ratio

        if encoder_cross_attention:
            self.encoder = Encoder_WithCrossAttention(seq_length_v=seq_length_v, seq_length_a=seq_length_a,
                                                      num_embeddings_v=num_embeddings_v, emb_dim_v=emb_dim_v,
                                                      num_indices_v=num_indices_v,
                                                      num_embeddings_a=num_embeddings_a, emb_dim_a=emb_dim_a,
                                                      num_indices_a=num_indices_a,
                                                      num_layer=encoder_num_layer, num_head=encoder_num_head,
                                                      alpha=alpha,
                                                      pos_embedding_trained=pos_embedding_trained,
                                                      vqvae_v_embedding=vqvae_v_embedding,
                                                      vqvae_a_embedding=vqvae_a_embedding, mlp_ratio=mlp_ratio)
        else:
            self.encoder = Encoder(seq_length_v=seq_length_v, seq_length_a=seq_length_a,
                                   num_embeddings_v=num_embeddings_v, emb_dim_v=emb_dim_v, num_indices_v=num_indices_v,
                                   num_embeddings_a=num_embeddings_a, emb_dim_a=emb_dim_a, num_indices_a=num_indices_a,
                                   num_layer=encoder_num_layer, num_head=encoder_num_head, alpha=alpha,
                                   pos_embedding_trained=pos_embedding_trained, vqvae_v_embedding=vqvae_v_embedding,
                                   vqvae_a_embedding=vqvae_a_embedding, mlp_ratio=mlp_ratio)

        if decoder_cross_attention:
            self.decoder = Decoder_WithCrossAttention(seq_length_v=seq_length_v, seq_length_a=seq_length_a,
                                                      emb_dim_v=emb_dim_v, num_indices_v=num_indices_v,
                                                      emb_dim_a=emb_dim_a, num_indices_a=num_indices_a,
                                                      num_layer=decoder_num_layer, num_head=decoder_num_head,
                                                      dim_tokens_v=dim_tokens_v,
                                                      dim_tokens_a=dim_tokens_a,
                                                      pos_embedding_trained=pos_embedding_trained, mlp_ratio=mlp_ratio)
        else:
            self.decoder = Decoder_WithoutCrossAttention(seq_length_v=seq_length_v, seq_length_a=seq_length_a,
                                                         emb_dim_v=emb_dim_v, num_indices_v=num_indices_v,
                                                         emb_dim_a=emb_dim_a, num_indices_a=num_indices_a,
                                                         num_layer=decoder_num_layer, num_head=decoder_num_head,
                                                         dim_tokens_v=dim_tokens_v,
                                                         dim_tokens_a=dim_tokens_a,
                                                         pos_embedding_trained=pos_embedding_trained,
                                                         mlp_ratio=mlp_ratio)

        if contrastive:
            self.pooling_visual = AttentionPooling(latent_dim=emb_dim_v * num_indices_v, num_heads=encoder_num_head,
                                                   mlp_ratio=mlp_ratio)
            self.pooling_audio = AttentionPooling(latent_dim=emb_dim_a * num_indices_a, num_heads=encoder_num_head,
                                                  mlp_ratio=mlp_ratio)
        self.contrastive = contrastive

    def forward(self, x_v, x_a, ratio=None):
        z_v, forward_indexes_v, backward_indexes_v, z_a, forward_indexes_a, backward_indexes_a = \
            self.encoder(x_v, x_a, ratio)
        x_v, mask_v, x_a, mask_a = self.decoder(z_v, backward_indexes_v, z_a, backward_indexes_a)
        if self.contrastive:
            cls_a = self.pooling_audio(z_a)
            cls_v = self.pooling_visual(z_v)
            return x_v, mask_v, cls_v, x_a, mask_a, cls_a
        else:
            return x_v, mask_v, x_a, mask_a

    def load(self, path_model: str):
        checkpoint = torch.load(path_model)
        state_dict = checkpoint["model"]
        self.load_state_dict(state_dict)
        loss = checkpoint['loss']
        print(f"\t [Model CMDMAE is loaded successfully with loss = {loss}]")


if __name__ == '__main__':
    v = torch.rand((25, 50, 10)).type(torch.LongTensor)
    a = torch.rand((25, 50, 10)).type(torch.LongTensor)
    cmd_mae = CMDMAE(seq_length_v=50,
                     seq_length_a=50,
                     emb_dim_v=4,
                     num_indices_v=10,
                     num_embeddings_v=128,
                     emb_dim_a=4,
                     num_indices_a=10,
                     num_embeddings_a=128,
                     encoder_num_head=2,
                     encoder_num_layer=1,
                     decoder_num_head=2,
                     decoder_num_layer=1,
                     dim_tokens_v=128,
                     dim_tokens_a=128,
                     alpha=(1.0, 1.0),
                     decoder_cross_attention=True,
                     encoder_cross_attention=False,
                     contrastive=True)
    # z_v, mask_v, z_a, mask_a = cmd_mae(v, a)
    z_v, mask_v, cls_v, z_a, mask_a, cls_a = cmd_mae(v, a)
    print(cls_v.shape)

