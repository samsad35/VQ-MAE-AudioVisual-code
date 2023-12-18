import torch
import torch.nn as nn
from .cmd_mae import CMDMAE
from einops import rearrange
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from .global_attention import AttentionPooling


class Query2Label(nn.Module):
    """Modified Query2Label model
    Unlike the model described in the paper (which uses a modified DETR
    transformer), this version uses a standard, unmodified Pytorch Transformer.
    Learnable label embeddings are passed to the decoder module as the target
    sequence (and ultimately is passed as the Query to MHA).
    """

    def __init__(
            self, cmdmae: CMDMAE, num_classes,
            nheads=8,
            encoder_layers=1,
            decoder_layers=2,
            use_pos_encoding=True):
        """Initializes model
        Args:
            model (str): Timm model descriptor for backbone.
            conv_out (int): Backbone output channels.
            num_classes (int): Number of possible label classes
            hidden_dim (int, optional): Hidden channels from linear projection of
            backbone output. Defaults to 256.
            nheads (int, optional): Number of MHA heads. Defaults to 8.
            encoder_layers (int, optional): Number of encoders. Defaults to 6.
            decoder_layers (int, optional): Number of decoders. Defaults to 6.
            use_pos_encoding (bool, optional): Flag for use of position encoding.
            Defaults to False.
        """

        super().__init__()
        self.contrastive = cmdmae.contrastive
        self.cross_encoder = cmdmae.encoder_cross_attention
        self.proj_v = cmdmae.encoder.proj_v
        self.proj_a = cmdmae.encoder.proj_a
        if not self.contrastive:
            self.pooling_visual = AttentionPooling(latent_dim=cmdmae.encoder.emb_dim_v * cmdmae.encoder.num_indices_v,
                                                   num_heads=cmdmae.encoder_num_head,
                                                   mlp_ratio=cmdmae.mlp_ratio)
            self.pooling_audio = AttentionPooling(latent_dim=cmdmae.encoder.emb_dim_a * cmdmae.encoder.num_indices_a,
                                                  num_heads=cmdmae.encoder_num_head,
                                                  mlp_ratio=cmdmae.mlp_ratio)
        else:
            self.pooling_visual = cmdmae.pooling_visual
            self.pooling_audio = cmdmae.pooling_audio
        self.pos_embedding_v = cmdmae.encoder.pos_embedding_v
        self.pos_embedding_a = cmdmae.encoder.pos_embedding_a
        self.modality_emb_v = cmdmae.encoder.modality_emb_v
        self.modality_emb_a = cmdmae.encoder.modality_emb_a
        self.pos_embedding_trained = cmdmae.encoder.pos_embedding_trained
        if self.cross_encoder:
            self.transformer_v = cmdmae.encoder.transformer_v
            self.layer_norm_v = cmdmae.encoder.layer_norm_v
            self.transformer_a = cmdmae.encoder.transformer_a
            self.layer_norm_a = cmdmae.encoder.layer_norm_a
            self.cross_attention_v = cmdmae.encoder.cross_attention_v
            self.cross_attention_a = cmdmae.encoder.cross_attention_a
        else:
            self.transformer = cmdmae.encoder.transformer
            self.layer_norm = cmdmae.encoder.layer_norm
        self.seq_length_v = cmdmae.encoder.seq_length_v
        self.seq_length_a = cmdmae.encoder.seq_length_a
        self.emb_dim_v = cmdmae.encoder.emb_dim_v
        self.num_indices_v = cmdmae.encoder.num_indices_v
        self.emb_dim_a = cmdmae.encoder.emb_dim_a
        self.num_indices_a = cmdmae.encoder.num_indices_a
        self.head = torch.nn.Linear(self.num_indices_a * self.emb_dim_a + self.num_indices_v * self.emb_dim_v,
                                    num_classes)
        self.num_classes = num_classes
        self.use_pos_encoding = use_pos_encoding
        self.hidden_dim = self.num_indices_a * self.emb_dim_a
        self.transformer_ = nn.Transformer(self.hidden_dim, nheads, encoder_layers, decoder_layers)

        if self.use_pos_encoding:
            # returns the encoding object
            self.pos_encoder = PositionalEncoding1D(self.hidden_dim)
            # returns the summing object
            self.encoding_adder = Summer(self.pos_encoder)

        # prediction head
        self.classifier = nn.Linear(num_classes * self.hidden_dim, num_classes)

        # learnable label embedding
        self.label_emb = nn.Parameter(torch.rand(1, num_classes, self.hidden_dim))

    def forward(self, x_v, x_a):
        """Passes batch through network
        Args:
            x (Tensor): Batch of images
        Returns:
            Tensor: Output of classification head
        """
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

        # Transfomer
        x_av = torch.cat((x_a, x_v), dim=0)
        if self.cross_encoder:
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

        else:
            x_av = rearrange(x_av, 't b c -> b t c')
            z_av = self.layer_norm(self.transformer(x_av))
            z_av = rearrange(z_av, 'b t c -> t b c')
            z_v = z_av[:x_v.shape[0]]
            z_a = z_av[x_v.shape[0]:]
        features = torch.cat((z_v, z_a), dim=0)
        features = rearrange(features, 't b c -> b t c')

        # add position encodings
        if self.use_pos_encoding:
            # input with encoding added
            features = self.encoding_adder(features)

        features = rearrange(features, 'b t c -> t b c')
        B = features.shape[1]

        # image feature vector "h" is sent in after transformation above; we
        # also convert label_emb from [1 x TARGET x (hidden)EMBED_SIZE] to
        # [TARGET x BATCH_SIZE x (hidden)EMBED_SIZE]
        label_emb = self.label_emb.repeat(B, 1, 1)
        label_emb = label_emb.transpose(0, 1)
        h = self.transformer_(features, label_emb).transpose(0, 1)

        # output from transformer was of dim [TARGET x BATCH_SIZE x EMBED_SIZE];
        # however, we transposed it to [BATCH_SIZE x TARGET x EMBED_SIZE] above.
        # below we reshape to [BATCH_SIZE x TARGET*EMBED_SIZE].
        #
        # next, we project transformer outputs to class labels
        h = torch.reshape(h, (B, self.num_classes * self.hidden_dim))

        return self.classifier(h)
