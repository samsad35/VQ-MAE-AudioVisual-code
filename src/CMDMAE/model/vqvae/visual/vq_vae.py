import torch
import torch.nn as nn
import torch.nn.functional as F
from .Encoder_Decoder import Encoder, Decoder
from .Vector_Quantizer import VectorQuantizer
from .Vector_Quantizer_EMA import VectorQuantizerEMA


class VQVAE(nn.Module):
    def __init__(self, num_hiddens=None,
                 num_residual_layers=None,
                 num_residual_hiddens=None,
                 num_embeddings=None,
                 embedding_dim=None,
                 commitment_cost=None,
                 decay=0,
                 input_channel=1):
        super(VQVAE, self).__init__()
        self.channels = input_channel
        self.encoder = Encoder(input_channel, num_hiddens,
                               num_residual_layers,
                               num_residual_hiddens)
        self.pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                     out_channels=embedding_dim,
                                     kernel_size=1,
                                     stride=1)
        if decay > 0.0:
            self.vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                             commitment_cost, decay)
        else:
            self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                          commitment_cost)
        self.decoder = Decoder(embedding_dim,
                               num_hiddens,
                               num_residual_layers,
                               num_residual_hiddens, out_channels=input_channel)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        loss, quantized, perplexity, _ = self.vq_vae(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity

    def load(self, path_model: str):
        checkpoint = torch.load(path_model)
        self.load_state_dict(checkpoint['model'])
        loss = checkpoint['loss']
        print(f"\t [Model VQ-VAE (visual) is loaded successfully with loss = {loss}]")

    def get_codebook_indices(self, images):
        z = self.encoder(images)
        z = self.pre_vq_conv(z)
        indices = self.vq_vae.get_codebook_indices(z)
        return indices

    def decode(self, indices):
        image_embeds = self.vq_vae.quantify(indices)
        images = self.decoder(image_embeds)
        return images

