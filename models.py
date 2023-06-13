import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm
from transformers import ViTImageProcessor, ViTModel
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms, utils, models

# ===========================================================#
# ===========================================================#
# NFNet
nfnet = timm.create_model('dm_nfnet_f0.dm_in1k', pretrained=False)
nfnet.load_state_dict(torch.load('./models/nfnet_best.pth.tar', map_location=torch.device('cpu'))['state_dict'])
nfnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# ===========================================================#
# ===========================================================#


# ===========================================================#
# ===========================================================#
# VQ-VAE
class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=(4, 4, 3, 1), stride=2):
        super(Encoder, self).__init__()
        
        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_size
        
        self.strided_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, stride, padding=1)
        self.strided_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, stride, padding=1)
        
        self.residual_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_3, padding=1)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, output_dim, kernel_4, padding=0)
        
    def forward(self, x):
        
        x = self.strided_conv_1(x)
        x = self.strided_conv_2(x)
        
        x = nn.functional.relu(x)
        y = self.residual_conv_1(x)
        y += x 
        
        x = nn.functional.relu(y)
        y = self.residual_conv_2(x)
        y += x
        
        return y  
    
class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                    torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = nn.functional.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))
    
    def retrieve_random_codebook(self, random_indices):
        quantized = nn.functional.embedding(random_indices, self.embedding)
        quantized = quantized.transpose(1, 3)
        
        return quantized

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)
        
        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = nn.functional.one_hot(indices, M).float()
        quantized = nn.functional.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        
        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        codebook_loss = nn.functional.mse_loss(x.detach(), quantized)
        e_latent_loss = nn.functional.mse_loss(x, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, commitment_loss, codebook_loss, perplexity


class Decoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_sizes=(1, 3, 2, 2), stride=2):
        super(Decoder, self).__init__()
        
        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_sizes
        
        self.residual_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, padding=0)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, padding=1)
        
        self.strided_t_conv_1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_3, stride, padding=0)
        self.strided_t_conv_2 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_4, stride, padding=0)
        
    def forward(self, x):
        
        y = self.residual_conv_1(x)
        y += x
        x = nn.functional.relu(y)
        
        y = self.residual_conv_2(x)
        y += x
        y = nn.functional.relu(y)
        
        y = self.strided_t_conv_1(y)
        y = self.strided_t_conv_2(y)
        
        return y
    
class VQVAE(nn.Module):
    def __init__(self, Encoder, Codebook, Decoder):
        super(VQVAE, self).__init__()
        self.encoder = Encoder
        self.codebook = Codebook
        self.decoder = Decoder
                
    def forward(self, x):
        z = self.encoder(x)
        z_quantized, commitment_loss, codebook_loss, perplexity = self.codebook(z)
        x_hat = self.decoder(z_quantized)
        
        return x_hat, commitment_loss, codebook_loss, perplexity
    

encoder = Encoder(input_dim=3, hidden_dim=64, output_dim=64)
codebook = VQEmbeddingEMA(n_embeddings=512, embedding_dim=64)
decoder = Decoder(input_dim=64, hidden_dim=64, output_dim=3)

vqvae = VQVAE(encoder, codebook, decoder)

vqvae.load_state_dict(torch.load('./models/VQ_VAE_128_5.pth.tar', map_location='cpu')['state_dict'])

vqvae_transform = transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor()
])
# ============================================================================= #
# ============================================================================= #

# ============================================================================= #
# ============================================================================= #
# Fine-Grain BCNN
features = 2048
fmap_size = 7

class BCNN(nn.Module):
    
    def __init__(self, fine_tune=False):
        
        super(BCNN, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        
        # freezing parameters
        if not fine_tune:
            
            for param in resnet.parameters():
                param.requires_grad = False
        else:
            
            for param in resnet.parameters():
                param.requires_grad = True

        layers = list(resnet.children())[:-2]
        self.features = nn.Sequential(*layers)        

        self.fc = nn.Linear(features ** 2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Initialize the fc layers.
        nn.init.xavier_normal_(self.fc.weight.data)
        
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)
        
    def forward(self, x):
        
        ## X: bs, 3, 256, 256
        ## N = bs
        N = x.size()[0]
        
        ## x : bs, 1024, 14, 14
        x = self.features(x)
        
        # bs, (1024 * 196) matmul (196 * 1024)
        x = x.view(N, features, fmap_size ** 2)
        x = self.dropout(x)
        
        # Batch matrix multiplication
        x = torch.bmm(x, torch.transpose(x, 1, 2))/ (fmap_size ** 2) 
        x = x.view(N, features ** 2)
        x = torch.sqrt(x + 1e-5)
        x = nn.functional.normalize(x)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

FG = BCNN(fine_tune=True)
FG.load_state_dict(torch.load('./models/best_bs128_10.pth', map_location='cpu')['state_dict'])

FG_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ============================================================================= #
# ============================================================================= #

# ============================================================================= #
# ============================================================================= #
# ViT
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', local_files_only=True)
vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', local_files_only=True)
vit.Classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 6)
)

vit.load_state_dict(torch.load('./models/checkpointsmodel1_epoch3.pth.tar', map_location='cpu')['model_state_dict'])

