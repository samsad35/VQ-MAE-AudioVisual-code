from .cmd_mae import CMDMAE
from .global_attention import AttentionPooling
import torch
from einops import rearrange


class Classifier(torch.nn.Module):
    def __init__(self, cmdmae: CMDMAE, num_classes=7, pooling: str = "attention"):
        super().__init__()
        self.pooling = pooling
        self.contrastive = cmdmae.contrastive
        self.cross_encoder = cmdmae.encoder_cross_attention
        self.proj_v = cmdmae.encoder.proj_v
        self.proj_a = cmdmae.encoder.proj_a
        if not self.contrastive and pooling == "attention":
            self.pooling_visual = AttentionPooling(latent_dim=cmdmae.encoder.emb_dim_v * cmdmae.encoder.num_indices_v,
                                                   num_heads=cmdmae.encoder_num_head,
                                                   mlp_ratio=cmdmae.mlp_ratio)
            self.pooling_audio = AttentionPooling(latent_dim=cmdmae.encoder.emb_dim_a * cmdmae.encoder.num_indices_a,
                                                  num_heads=cmdmae.encoder_num_head,
                                                  mlp_ratio=cmdmae.mlp_ratio)
        elif not self.contrastive and pooling == "mean":
            pass
        elif self.contrastive:
            self.pooling_visual = cmdmae.pooling_visual
            self.pooling_audio = cmdmae.pooling_audio
        self.pos_embedding_v = cmdmae.encoder.pos_embedding_v
        self.pos_embedding_a = cmdmae.encoder.pos_embedding_a
        self.modality_emb_v = cmdmae.encoder.modality_emb_v
        self.modality_emb_a = cmdmae.encoder.modality_emb_a
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
        self.pos_embedding_trained = cmdmae.encoder.pos_embedding_trained

    def forward(self, x_v, x_a, return_attention=False):
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
        if self.cross_encoder:
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

        else:
            x_av = torch.cat((x_v, x_a), dim=0)
            x_av = rearrange(x_av, 't b c -> b t c')
            z_av = self.layer_norm(self.transformer(x_av))
            z_av = rearrange(z_av, 'b t c -> t b c')
            z_v = z_av[:x_v.shape[0]]
            z_a = z_av[x_v.shape[0]:]
        if self.pooling == "attention":
            if return_attention:
                cls_a, att_a = self.pooling_audio(z_a, return_attention=return_attention)
                cls_v, att_v = self.pooling_visual(z_v, return_attention=return_attention)
                cls_a = cls_a[0]
                cls_v = cls_v[0]
            else:
                cls_a = self.pooling_audio(z_a, return_attention=return_attention)[0]
                cls_v = self.pooling_visual(z_v, return_attention=return_attention)[0]
        else:
            cls_a = torch.mean(z_a, dim=0)
            cls_v = torch.mean(z_v, dim=0)
        # Classifier
        logits = self.head(torch.cat((cls_a, cls_v), dim=-1))
        if return_attention:
            return logits, att_a, att_v
        else:
            return logits

    def load(self, path_model: str):
        checkpoint = torch.load(path_model)
        state_dict = checkpoint["model"]
        self.load_state_dict(state_dict)
        loss = checkpoint['loss']
        print(f"\t [Model CMDMAE Classifier is loaded successfully with loss = {loss}]")


    # def get_cls(self, patches):
    #     patches = rearrange(patches, 'b t c -> t b c')
    #     patches = self.proj(patches).reshape(self.seq_length, -1, self.dim_indices * self.emb_dim)
    #     patches = patches + self.pos_embedding
    #     # patches = self.pos_embedding(patches)
    #     patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
    #     patches = rearrange(patches, 't b c -> b t c')
    #     features = self.layer_norm(self.transformer(patches))
    #     features = rearrange(features, 'b t c -> t b c')
    #     cls = features[0]
    #     return cls
