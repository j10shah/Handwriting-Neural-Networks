# Name: Jay Shah
# Date: 03/31/2026
# Purpose: Task 4 - Define a transformer-based neural network for MNIST
#          classification using patch embeddings and transformer encoder layers.
# Run: Imported as a module in training script

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetConfig: # Configuration class to hold hyperparameters and settings for the transformer-based model, including patch size, embedding dimension, number of layers, heads, MLP dimension, dropout rate, training parameters, and device configuration
    def __init__( 
        self,
        name='vit_base',
        dataset='mnist',
        patch_size=7,
        stride=7,
        embed_dim=48,
        depth=4,
        num_heads=8,
        mlp_dim=128,
        dropout=0.1,
        use_cls_token=False,
        epochs=15,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        seed=0,
        optimizer='adamw',
        device='cpu',
    ): # Initialize the configuration with default values for the transformer-based model, allowing for customization of various hyperparameters and training settings to optimize performance on the MNIST classification task
        self.image_size = 28
        self.in_channels = 1
        self.num_classes = 10

        self.name = name
        self.dataset = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.use_cls_token = use_cls_token
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.optimizer = optimizer
        self.device = device



class PatchEmbedding(nn.Module): # Module to convert input images into patch embeddings by extracting patches using unfold, projecting them to a specified embedding dimension, and adding positional embeddings for use in the transformer encoder layers
    def __init__(self, image_size, patch_size, stride, in_channels, embed_dim): # Initialize the patch embedding module with parameters for image size, patch size, stride, input channels, and embedding dimension, and set up the necessary layers for unfolding the input images into patches and projecting them to the desired embedding dimension
        super().__init__()

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)

        positions_per_dim = ((image_size - patch_size) // stride) + 1
        self.num_patches = positions_per_dim * positions_per_dim

    def forward(self, x): # Forward pass to convert input images into patch embeddings by applying the unfold operation to extract patches, transposing the resulting tensor to have the correct shape for projection, and then applying a linear projection to obtain the final patch embeddings for use in the transformer encoder layers
        x = self.unfold(x)     
        x = x.transpose(1, 2)    
        x = self.proj(x)    
        return x



class NetTransformer(nn.Module): # Transformer-based neural network architecture for MNIST classification, consisting of a patch embedding layer, optional CLS token, positional embeddings, transformer encoder layers, and a final classifier to output class probabilities
    def __init__(self, config): # Initialize the transformer-based model with the provided configuration, setting up the patch embedding layer, optional CLS token, positional embeddings, transformer encoder layers, layer normalization, and classifier to create a complete architecture for MNIST classification using a vision transformer approach
        super(NetTransformer, self).__init__()

        self.patch_embed = PatchEmbedding( # Initialize the patch embedding layer using the parameters specified in the configuration, which will convert input images into a sequence of patch embeddings for processing by the transformer encoder layers
            config.image_size,
            config.patch_size,
            config.stride,
            config.in_channels,
            config.embed_dim
        )

        num_tokens = self.patch_embed.num_patches
        print("Number of tokens:", num_tokens)

        self.use_cls_token = config.use_cls_token

        if self.use_cls_token: # If using a CLS token, initialize it as a learnable parameter and adjust the total number of tokens to account for the additional CLS token, which will be used to aggregate information from the patch embeddings for classification in the transformer architecture
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            total_tokens = num_tokens + 1
        else: # If not using a CLS token, set the CLS token to None and keep the total number of tokens equal to the number of patch embeddings, which will be processed by the transformer encoder layers without an additional token for classification
            self.cls_token = None
            total_tokens = num_tokens

        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, config.embed_dim))
        self.pos_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer( # Initialize a single transformer encoder layer with the specified embedding dimension, number of heads, feedforward dimension, dropout rate, activation function, batch-first configuration, and normalization settings to be used in the transformer encoder stack for processing the patch embeddings and learning complex representations for MNIST classification
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder( # Stack multiple transformer encoder layers as specified in the configuration to create a deep transformer architecture that can effectively learn from the patch embeddings and positional information for accurate MNIST classification
            encoder_layer,
            num_layers=config.depth
        )

        self.norm = nn.LayerNorm(config.embed_dim)

        self.classifier = nn.Sequential( # Define the classifier as a sequential model consisting of a linear layer to project the final transformer output to an intermediate dimension, followed by a GELU activation, and another linear layer to project to the number of classes for MNIST classification, with a log softmax activation at the end to output class probabilities
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.num_classes)
        )

    def forward(self, x): # Define the forward pass through the
        x = self.patch_embed(x)   # (B, N, D)

        batch_size = x.size(0)

        if self.use_cls_token: # If using a CLS token, expand it to match the batch size and concatenate it to the beginning of the patch embeddings, then add positional embeddings to the combined sequence of tokens before passing it through the transformer encoder layers for processing and classification in the vision transformer architecture
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        x = x + self.pos_embed

        x = self.pos_dropout(x)

        x = self.encoder(x)

        if self.use_cls_token: # If using a CLS token, extract the output corresponding to the CLS token from the transformer encoder output to be used for classification, as it is designed to aggregate information from all the patch embeddings for making the final class prediction in the transformer architecture
            x = x[:, 0]   # CLS token
        else: # If not using a CLS token, average the output across all tokens from the transformer encoder to create a single representation for classification, which will be passed through the final classifier to output class probabilities for MNIST classification
            x = x.mean(dim=1)

        x = self.norm(x)

        x = self.classifier(x)

        return F.log_softmax(x, dim=1)