import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self,dim_input, dim_hiddens, dim_residual_hiddens):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim_input,dim_residual_hiddens,3,1,1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(dim_residual_hiddens,dim_hiddens,1, bias= False),
        )

    def forward(self,x):

        return x + self.block(x)

class ResidualBlocks(nn.Module):
    def __init__(self,dim_input, dim_hiddens, dim_residual_hiddens, num_residual_layers):
        super(ResidualBlocks, self).__init__()

        self.num_residual_layers = num_residual_layers
        self.blocks = nn.ModuleList(
            [ResidualBlock(dim_input,dim_hiddens,dim_residual_hiddens)] * num_residual_layers
        )

    def forward(self, x):

        for i in range(self.num_residual_layers):
            x = self.blocks[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, dim, residual_channels,n_layers, embedding_d):
        super(Encoder, self).__init__()

        input_dim = 3
        self.conv1 = nn.Conv2d(input_dim, dim // 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(dim // 2, dim, 4, 2, 1)
        self.conv3 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.residual_blocks = ResidualBlocks(dim, dim, residual_channels, n_layers)
        self.conv4 = nn.Conv2d(dim, embedding_d, 1, 1)

    def forward(self, x):

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.conv3(h)
        h = self.residual_blocks(h)
        z = self.conv4(h)

        return z

class Decoder(nn.Module):
    def __init__(self, dim, residual_channels,n_layers, embedding_d):
        super(Decoder, self).__init__()

        input_dim = 3
        self.conv1 = nn.Conv2d(embedding_d, dim, 3, 1, 1)
        self.residual_blocks = ResidualBlocks(dim, dim, residual_channels, n_layers)
        self.upconv1 = nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1)
        self.upconv2 = nn.ConvTranspose2d(dim // 2, input_dim, 4, 2, 1)

    def forward(self, x):

        h = self.conv1(x)
        h = self.residual_blocks(h)
        h = F.relu(self.upconv1(h))
        y = self.upconv2(h)

        return y

class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq):

        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq

    def forward(self, x):

        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.vq(z)
        x_recon = self.decoder(quantized)

        return loss, x_recon, perplexity

class SpeechEncoder(nn.Module):
    def __init__(self, dim):  # dim == d
        super(SpeechEncoder, self).__init__()

        self.dim = dim
        self.conv1 = nn.Conv2d(1, dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.conv6 = nn.Conv2d(dim, dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x):

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        z = self.conv6(h)

        return z

class ConditionEmbedding(nn.Module):
    def __init__(self, num_global_condition, global_condition_dim, local_condition_dim, upscale_factor = 64):
        super(ConditionEmbedding, self).__init__()

        self.local_embed1 = nn.Conv2d(local_condition_dim, local_condition_dim, kernel_size=(3, 1), padding=(1, 0),
                                      dilation=(1, 1))
        self.local_embed2 = nn.Conv2d(local_condition_dim, local_condition_dim, kernel_size=(3, 1), padding=(2, 0),
                                      dilation=(2, 1))
        self.local_embed3 = nn.Conv2d(local_condition_dim, local_condition_dim, kernel_size=(3, 1), padding=(4, 0),
                                      dilation=(4, 1))
        self.local_embed4 = nn.Conv2d(local_condition_dim, local_condition_dim, kernel_size=(3, 1), padding=(8, 0),
                                      dilation=(8, 1))
        self.local_embed5 = nn.Conv2d(local_condition_dim, local_condition_dim, kernel_size=(3, 1), padding=(16, 0),
                                      dilation=(16, 1))

        self.upscale_factor = upscale_factor

        self.global_embed = nn.Embedding(num_global_condition, global_condition_dim)

    def forward(self, local_condition, global_condition):

        local_condition = F.relu(self.local_embed1(local_condition))
        local_condition = F.relu(self.local_embed2(local_condition))
        local_condition = F.relu(self.local_embed3(local_condition))
        local_condition = F.relu(self.local_embed4(local_condition))
        local_condition = F.relu(self.local_embed5(local_condition))
        local_condition = F.interpolate(local_condition, torch.Size((self.upscale_factor * local_condition.shape[2], 1)))

        global_condition = self.global_embed(global_condition)
        global_condition = global_condition.unsqueeze(2).unsqueeze(2)

        global_condition = F.interpolate(
            global_condition, torch.Size((local_condition.shape[2], 1)))

        condition = torch.cat([local_condition, global_condition], dim=1)

        return condition

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_k, embedding_d, beta, decay = None, dtype = torch.cuda.FloatTensor, epsilon = 1e-5):
        super(VectorQuantizer, self).__init__()

        self.embedding_k = embedding_k #K: the size of discrete latent space
        self.embedding_d = embedding_d  #D: the dimensionality of each latent embedding vector e_i
        self.beta = beta #beta(0.25) range[0.1,2.0] in eq. (3)

        self.embedding = nn.Embedding(embedding_k, embedding_d) #[K,D]
        self.embedding.weight.data.uniform_(-1./embedding_k,1./embedding_k)

        self.dtype = dtype

        #for EMA
        self.register_buffer('ema_cluster_size', torch.zeros(embedding_k))
        self.ema_w = nn.Parameter(torch.Tensor(embedding_k, self.embedding_d))
        self.ema_w.data.normal_()
        self.decay = decay
        self.epsilon = epsilon

    def forward(self, inputs):

        inputs = inputs.permute(0, 2, 3, 1).contiguous() #[B, C, H, W] => [B, H, W, C]
        input_shape = inputs.shape

        flattened_input = inputs.view(-1,self.embedding_d) #[B, H, W, C] => [B x H x W , D = C]

        distances = (torch.sum(flattened_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flattened_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.embedding_k).type(self.dtype)
        encodings.scatter_(1, encoding_indices, 1)

        # Update the embedding vectors using EMA
        if self.training == True and self.decay != None:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1.0 - self.decay) * torch.sum(encodings,0)

            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = ( (self.ema_cluster_size + self.epsilon) / (n + self.embedding_k * self.epsilon) * n)

            dw = torch.matmul(encodings.t(), flattened_input)

            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)

            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))


        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)


        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        loss = q_latent_loss + self.beta * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class SpeechVQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq):
        super(SpeechVQVAE, self).__init__()
        self.encoder = encoder
        self.vq = vq
        self.decoder = decoder
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x_enc, x_dec, global_condition, t):
        z = self.encoder(x_enc)
        loss2, quantized, perplexity, _ = self.vq(z)
        local_condition = quantized
        local_condition = local_condition.squeeze(-1)
        x_dec = x_dec.squeeze(-1)
        y = self.decoder(x_dec,local_condition,global_condition)
        y = y.unsqueeze(-1)
        loss1 = self.loss_func(y,t)
        loss = loss1+ loss2

        return loss, y




