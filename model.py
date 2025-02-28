import torch
import torch.nn as nn
import torch.nn.functional as F
from deepd3_model import EncoderBlock, Decoder, DecoderBlock

class HPObCustomUNet(nn.Module):
    """
    A custom UNet that integrates a latent variational bottleneck (from HPob-UNet)
    into the DeepD3Model architecture. It provides two separate outputs (e.g. dendrites and spines)
    and computes a KL divergence term when a target (y) is provided.
    """
    def __init__(self, in_channels=1, base_filters=32, num_layers=4, 
                 activation="swish", use_batchnorm=True, apply_last_layer=True):
        super().__init__()
        self.apply_last_layer = apply_last_layer
        # Choose activation function
        if activation == "swish":
            act = nn.SiLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        self.activation = act
        self.num_layers = num_layers
        self.base_filters = base_filters
        self.use_batchnorm = use_batchnorm
        
        # ---------------------------
        # Encoder (same as in DeepD3Model)
        # ---------------------------
        self.encoder_blocks = nn.ModuleList()
        current_in = in_channels
        for i in range(num_layers):
            out_channels = base_filters * (2 ** i)
            self.encoder_blocks.append(EncoderBlock(current_in, out_channels, self.activation, use_batchnorm))
            current_in = out_channels  # pooling does not change channel count

        # ---------------------------
        # Latent Bottleneck
        # ---------------------------
        # Process the deepest encoder output with a latent conv block.
        latent_in = base_filters * (2 ** (num_layers - 1))
        latent_out = base_filters * (2 ** num_layers)
        self.latent_conv = nn.Sequential(
            nn.Conv2d(latent_in, latent_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_out) if use_batchnorm else nn.Identity(),
            self.activation
        )
        # Compute prior distribution parameters (mean and log_std) from the latent features.
        self.prior_mean_conv = nn.Conv2d(latent_out, latent_out, kernel_size=1)
        self.prior_log_std_conv = nn.Conv2d(latent_out, latent_out, kernel_size=1)
        
        # ---------------------------
        # Posterior Branch (optional)
        # ---------------------------
        # If a target (y) is provided, we build a separate encoder branch.
        # Here we assume y has 1 channel; adjust if needed.
        self.posterior_encoder_blocks = nn.ModuleList()
        current_in_post = in_channels + 1  # Concatenate x and y along channel dimension.
        for i in range(num_layers):
            out_channels = base_filters * (2 ** i)
            self.posterior_encoder_blocks.append(EncoderBlock(current_in_post, out_channels, self.activation, use_batchnorm))
            current_in_post = out_channels
        self.posterior_latent_conv = nn.Sequential(
            nn.Conv2d(base_filters * (2 ** (num_layers - 1)), latent_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_out) if use_batchnorm else nn.Identity(),
            self.activation
        )
        self.post_mean_conv = nn.Conv2d(latent_out, latent_out, kernel_size=1)
        self.post_log_std_conv = nn.Conv2d(latent_out, latent_out, kernel_size=1)
        
        # ---------------------------
        # Decoders (for two separate outputs)
        # ---------------------------
        self.decoder_dendrites = Decoder(num_layers, base_filters, self.activation, use_batchnorm)
        self.decoder_spines = Decoder(num_layers, base_filters, self.activation, use_batchnorm)
        
    def reparameterize(self, mean, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def forward(self, x, y=None):
        # ---------- Encoder ----------
        encoder_features = []
        out = x
        for block in self.encoder_blocks:
            feat, out = block(out)
            encoder_features.append(feat)
            
        # ---------- Latent Bottleneck ----------
        latent_feature = self.latent_conv(out)
        prior_mean = self.prior_mean_conv(latent_feature)
        prior_log_std = self.prior_log_std_conv(latent_feature)
        prior_latent = self.reparameterize(prior_mean, prior_log_std)
        
        kl_div = None
        # If target y is provided, compute posterior latent and KL divergence.
        if y is not None:
            # Build posterior input by concatenating x and y.
            posterior_input = torch.cat([x, y], dim=1)
            post_out = posterior_input
            for block in self.posterior_encoder_blocks:
                feat, post_out = block(post_out)
            post_latent_feature = self.posterior_latent_conv(post_out)
            post_mean = self.post_mean_conv(post_latent_feature)
            post_log_std = self.post_log_std_conv(post_latent_feature)
            post_latent = self.reparameterize(post_mean, post_log_std)
            
            # For decoding, use the posterior latent.
            latent_feature = post_latent
            
            # Compute KL divergence between posterior and prior distributions.
            prior_std = torch.exp(prior_log_std)
            post_std = torch.exp(post_log_std)
            kl_element = (torch.log(prior_std / post_std) + 
                          (post_std.pow(2) + (post_mean - prior_mean).pow(2)) / (2 * prior_std.pow(2)) - 0.5)
            # Sum over all dimensions except the batch dimension.
            kl_div = kl_element.view(kl_element.size(0), -1).sum(dim=1).mean()
        else:
            # If no target provided, you could use the prior latent.
            latent_feature = prior_latent
        
        # ---------- Decoding ----------
        # Pass copies of encoder features to each decoder.
        encoder_features_d = encoder_features.copy()
        encoder_features_s = encoder_features.copy()
        dendrites = self.decoder_dendrites(latent_feature, encoder_features_d)
        spines = self.decoder_spines(latent_feature, encoder_features_s)
        
        if self.apply_last_layer:
            return dendrites, spines, kl_div
        else:
            # Return intermediate features if desired.
            return (self.decoder_dendrites.last_layer_features, 
                    self.decoder_spines.last_layer_features, 
                    kl_div)

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == '__main__':
    # Create a dummy input image and a target (e.g. segmentation map)
    x = torch.randn(1, 1, 128, 128)   # Input image
    y = torch.randn(1, 1, 128, 128)   # Target (for posterior branch)
    
    model = HPObCustomUNet(in_channels=1, base_filters=32, num_layers=4, activation="swish")
    dendrites, spines, kl = model(x, y)
    print("Dendrites output shape:", dendrites.shape)
    print("Spines output shape:", spines.shape)
    if kl is not None:
        print("KL divergence:", kl.item())
