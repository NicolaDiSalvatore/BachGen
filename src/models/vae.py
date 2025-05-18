"""
This model is designed to learn musical sequences in an unsupervised way using a VAE architecture.
It consists of:
- An LSTM Encoder: Encodes the input sequence into a latent representation.
- A Latent Layer: Learns the mean and log variance for the latent distribution.
- A Reparameterization Trick: Allows sampling from the latent space.
- An LSTM Decoder: Reconstructs the sequence from the latent vector.
- A Loss Function: Combines reconstruction loss and KL divergence.

Inputs:
- x: A batch of sequences with shape (batch_size, seq_len, input_dim)

Outputs:
- Reconstructed sequences, and latent variables (mean, logvar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_layers=1):
        super(SequenceVAE, self).__init__()
        self.input_dim = input_dim          # Dimensionality of each input token
        self.latent_dim = latent_dim        # Size of the latent space
        self.hidden_dim = hidden_dim        # Hidden size for LSTM
        self.num_layers = num_layers        # Number of layers in LSTM

        
        # Encoder: LSTM + Linear layers for mean and logvar
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: Linear to hidden -> LSTM -> Linear to output
        self.fc_z_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        Encodes input sequences to latent space.
        Returns the mean and log variance of the latent distribution.
        """
        _, (h_n, _) = self.encoder_lstm(x)   # h_n: (num_layers, batch, hidden_dim)
        h = h_n[-1]                          # Take the last layer's hidden state
        mean = self.fc_mean(h)               # Latent mean
        logvar = self.fc_logvar(h)           # Latent log-variance
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick: z = mean + std * epsilon
        Allows gradients to pass through the stochastic node.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, seq_len):
        """
        Decodes latent vector z into a sequence of length seq_len.
        """
        batch_size = z.size(0)

        # Project z to initial hidden state for the decoder
        h_0 = self.fc_z_to_hidden(z).view(self.num_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros_like(h_0)  # Initial cell state

        # Use zero vectors as initial input to the decoder
        decoder_input = torch.zeros(batch_size, seq_len, self.input_dim, device=z.device)

        # Decode the sequence
        output, _ = self.decoder_lstm(decoder_input, (h_0, c_0))
        return self.decoder_out(output)  # Project hidden states to output space

    def forward(self, x):
        """
        Full forward pass: encode -> reparameterize -> decode.
        """
        seq_len = x.size(1)
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z, seq_len)
        return recon_x, mean, logvar

    def loss_function(self, recon_x, x, mean, logvar):
        """
        VAE loss function:
        - Reconstruction loss (Binary Cross-Entropy or MSE)
        - KL Divergence (regularization)
        """
        # Binary cross-entropy works best when output is in [0,1]
        recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='mean')
        kl_div = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kl_div, recon_loss, kl_div