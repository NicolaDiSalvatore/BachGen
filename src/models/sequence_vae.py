"""
Improved Sequential VAE for Musical Data

This enhanced model includes:
- Teacher forcing during training and autoregressive generation during inference
- Bidirectional encoder for better context understanding
- Better cell state initialization from latent vector
- Dropout regularization to prevent overfitting
- Flexible loss functions for different data types
- Beta-VAE support for controlling disentanglement

Key improvements over the original:
1. Proper sequential conditioning (teacher forcing/autoregressive)
2. Richer encoding with bidirectional LSTM
3. Better information flow through improved initialization
4. Regularization and flexible training options
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedSequenceVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_layers=1,
                 dropout=0.2, loss_type='mse', bidirectional=True):
        super(ImprovedSequenceVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.loss_type = loss_type
        self.bidirectional = bidirectional

        # Encoder: Bidirectional LSTM + Linear layers for mean and logvar
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # Account for bidirectional hidden size
        encoder_hidden_size = hidden_dim * (2 if bidirectional else 1)

        self.fc_mean = nn.Linear(encoder_hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(encoder_hidden_size, latent_dim)

        # Decoder components
        # Project latent vector to both hidden and cell states
        self.fc_z_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_z_to_cell = nn.Linear(latent_dim, hidden_dim * num_layers)

        # Decoder LSTM (unidirectional for generation)
        self.decoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.decoder_out = nn.Linear(hidden_dim, input_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        """
        Encodes input sequences to latent space using bidirectional LSTM.
        Returns the mean and log variance of the latent distribution.
        """
        _, (h_n, _) = self.encoder_lstm(x)

        if self.bidirectional:
            # Concatenate forward and backward final states
            # h_n shape: [num_layers * 2, batch, hidden_dim]
            h_forward = h_n[-2]  # Last layer, forward direction
            h_backward = h_n[-1]  # Last layer, backward direction
            h = torch.cat([h_forward, h_backward], dim=1)
        else:
            h = h_n[-1]  # Take the last layer's hidden state

        # Apply dropout before final projection
        h = self.dropout(h)

        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick: z = mean + std * epsilon
        Allows gradients to pass through the stochastic node.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode_with_teacher_forcing(self, z, target_sequence):
        """
        Decodes latent vector z using teacher forcing.
        Uses ground truth sequence shifted by one position as input.
        """
        batch_size = z.size(0)

        # Initialize both hidden and cell states from latent vector
        h_0 = self.fc_z_to_hidden(z).view(self.num_layers, batch_size, self.hidden_dim)
        c_0 = self.fc_z_to_cell(z).view(self.num_layers, batch_size, self.hidden_dim)

        # Create decoder input: [START_TOKEN, x1, x2, ..., x_{n-1}]
        start_token = torch.zeros(batch_size, 1, self.input_dim, device=z.device)
        decoder_input = torch.cat([start_token, target_sequence[:, :-1, :]], dim=1)

        # Run LSTM decoder with teacher forcing
        output, _ = self.decoder_lstm(decoder_input, (h_0, c_0))

        # Project to output space
        return self.decoder_out(output)

    def decode_autoregressive(self, z, seq_len):
        """
        Decodes latent vector z autoregressively.
        Feeds model's own outputs as inputs for next timestep.
        """
        batch_size = z.size(0)

        # Initialize both hidden and cell states from latent vector
        h_0 = self.fc_z_to_hidden(z).view(self.num_layers, batch_size, self.hidden_dim)
        c_0 = self.fc_z_to_cell(z).view(self.num_layers, batch_size, self.hidden_dim)
        hidden = (h_0, c_0)

        # Start with zero/start token
        current_input = torch.zeros(batch_size, 1, self.input_dim, device=z.device)
        outputs = []

        # Generate sequence step by step
        for t in range(seq_len):
            output, hidden = self.decoder_lstm(current_input, hidden)
            output = self.decoder_out(output)
            outputs.append(output)

            # Use current output as next input (autoregressive)
            current_input = output

        return torch.cat(outputs, dim=1)

    def decode(self, z, seq_len, target_sequence=None):
        """
        Unified decode function that chooses method based on training mode.
        """
        if self.training and target_sequence is not None:
            return self.decode_with_teacher_forcing(z, target_sequence)
        else:
            return self.decode_autoregressive(z, seq_len)

    def forward(self, x, use_teacher_forcing=None):
        """
        Full forward pass: encode -> reparameterize -> decode.
        """
        seq_len = x.size(1)
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)

        # Determine decoding method
        if use_teacher_forcing is None:
            use_teacher_forcing = self.training

        if use_teacher_forcing:
            recon_x = self.decode_with_teacher_forcing(z, x)
        else:
            recon_x = self.decode_autoregressive(z, seq_len)

        return recon_x, mean, logvar, z

    def loss_function(self, recon_x, x, mean, logvar, beta=1.0):
        """
        Improved VAE loss function with flexible reconstruction loss.
        
        Args:
            recon_x: Reconstructed sequences
            x: Original sequences  
            mean: Latent mean
            logvar: Latent log variance
            beta: Weight for KL divergence (beta-VAE)
        
        Returns:
            total_loss, reconstruction_loss, kl_divergence
        """
        # Flexible reconstruction loss based on data type
        if self.loss_type == 'bce':
            recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='mean')
        elif self.loss_type == 'mse':
            recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        elif self.loss_type == 'l1':
            recon_loss = F.l1_loss(recon_x, x, reduction='mean')
        elif self.loss_type == 'huber':
            recon_loss = F.huber_loss(recon_x, x, reduction='mean')
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        # KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0,I)
        kl_div = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        # Total loss with beta weighting
        total_loss = recon_loss + beta * kl_div

        return total_loss, recon_loss, kl_div

    def generate(self, num_samples=1, seq_len=None, z=None):
        """
        Generate new sequences by sampling from latent space.
        
        Args:
            num_samples: Number of sequences to generate
            seq_len: Length of sequences to generate
            z: Optional latent vectors (if None, samples from N(0,I))
        
        Returns:
            Generated sequences
        """
        self.eval()

        with torch.no_grad():
            if z is None:
                # Sample from prior N(0,I)
                z = torch.randn(num_samples, self.latent_dim)

            if seq_len is None:
                seq_len = 32  # Default sequence length

            # Generate sequences
            generated = self.decode_autoregressive(z, seq_len)

        return generated

    def interpolate(self, x1, x2, num_steps=10):
        """
        Interpolate between two input sequences in latent space.
        
        Args:
            x1, x2: Input sequences to interpolate between
            num_steps: Number of interpolation steps
        
        Returns:
            Interpolated sequences
        """
        self.eval()

        with torch.no_grad():
            # Encode both sequences
            mean1, _ = self.encode(x1.unsqueeze(0))
            mean2, _ = self.encode(x2.unsqueeze(0))

            # Create interpolation weights
            alphas = torch.linspace(0, 1, num_steps).view(-1, 1)

            # Interpolate in latent space
            z_interp = alphas * mean2 + (1 - alphas) * mean1

            # Decode interpolated latent vectors
            seq_len = x1.size(0)
            interpolated = self.decode_autoregressive(z_interp, seq_len)

        return interpolated



# Example beta annealing schedule
def beta_annealing_schedule(epoch, start_beta=0.0, end_beta=1.0, anneal_epochs=50):
    """Linear beta annealing to prevent posterior collapse."""
    if epoch < anneal_epochs:
        return start_beta + (end_beta - start_beta) * (epoch / anneal_epochs)
    return end_beta

# Example model instantiation
if __name__ == "__main__":
    # Model parameters
    input_dim = 88      # Piano keys (or your musical representation dimension)
    latent_dim = 32     # Latent space dimension
    hidden_dim = 256    # LSTM hidden dimension
    num_layers = 2      # Number of LSTM layers

    # Create improved model
    model = ImprovedSequenceVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.2,
        loss_type='mse',  # or 'bce' depending on your data
        bidirectional=True
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Encoder is {'bidirectional' if model.bidirectional else 'unidirectional'}")
    print(f"Using {model.loss_type} reconstruction loss")
