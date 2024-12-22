from PIL import Image
from glob import glob
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    def __init__(self, path, transform=None):
        self.images = glob(f"{path}/*.jpg")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx % len(self.images)]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

def get_transform(image_size):
    transform = T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform

class MappingNetwork(nn.Module):
    def __init__(self, latent, nlayers):
        super(MappingNetwork, self).__init__()
        layers = []
        for _ in range(nlayers):
            layers.append(nn.Linear(latent, latent))
            layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x / x.norm(dim=1, keepdim=True)
        return self.layers(x)

class AdaINBlock(nn.Module):
    def __init__(self, nc):
        super(AdaINBlock, self).__init__()
        self.norm = nn.InstanceNorm2d(nc)
        self.style_transform = nn.LazyLinear(nc * 2)

    def forward(self, x, style):
        style = self.style_transform(style)
        style = style.view(style.size(0), -1, 1, 1)
        ys, yb = style.chunk(2, dim=1)
        x = self.norm(x)
        return ys * x + yb

class SynthesisBlock(nn.Module):
    def __init__(self, nc):
        super(SynthesisBlock, self).__init__()
        self.noise1 = nn.Parameter(torch.randn(1, nc, 1, 1))
        self.conv1 = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        self.adain1 = AdaINBlock(nc)

        self.noise2 = nn.Parameter(torch.randn(1, nc, 1, 1))
        self.conv2 = nn.Conv2d(nc, nc // 2, kernel_size=3, padding=1)
        self.adain2 = AdaINBlock(nc // 2)


    def forward(self, x, w, skip_conv1=False):
        noise1 = self.noise1 * torch.randn_like(x)
        x = x + noise1
        if not skip_conv1:
            x = self.conv1(x)
        x = self.adain1(x, w)

        noise2 = self.noise2 * torch.randn_like(x)
        x = x + noise2
        x = self.conv2(x)
        x = self.adain2(x, w)
        return x

class GeneratorNetwork(nn.Module):
    def __init__(self, latent, image_size):
        super(GeneratorNetwork, self).__init__()
        self.n_synth = int(np.log2(image_size)) - 3
        self.mapping = MappingNetwork(latent, self.n_synth)
        synth_layers = []
        nc = image_size
        for i in range(self.n_synth):
            synth_layers.append(SynthesisBlock(nc))
            synth_layers.append(
                nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            synth_layers.append(nn.LeakyReLU(0.2))
            nc //= 2
        self.constant = nn.Parameter(torch.randn(1, image_size, 8, 8))
        self.synthesis = nn.ModuleList(synth_layers)
        self.make_img = nn.Conv2d(nc, 3, kernel_size=3, padding=1)

    def forward(self, latent):
        latent = F.normalize(latent, p=2.0, dim=1)
        w = self.mapping(latent)
        x = self.constant.repeat(latent.size(0), 1, 1, 1)
        for i in range(0, 3 * self.n_synth, 3):
            x = self.synthesis[i](
                x, w, skip_conv1=True if i == 0 else False
            )  # SynthesisBlock
            # Based on paper, the first synthesis block directly uses the
            # const image, and does not use convolution
            x = self.synthesis[i + 1](x)  # Upsample
            x = self.synthesis[i + 2](x)  # activation
        return self.make_img(x)

class DiscriminatorNetwork(nn.Module):
    def __init__(self, image_size):
        super(DiscriminatorNetwork, self).__init__()
        ncs = 16
        layers = [
            nn.Conv2d(3, ncs, kernel_size=3),
            nn.LeakyReLU(0.2),
        ]
        out = image_size - 3 + 1
        for i in range(5):
            layers.append(nn.Conv2d(ncs, ncs * 2, kernel_size=3, stride=2))
            layers.append(nn.LeakyReLU(0.2))
            if i > 0:
                layers.append(nn.Dropout(p=i/20))
            ncs = ncs * 2
            out = (out - 3 + 1) // 2
        layers.append(nn.Flatten())
        layers.append(nn.Dropout(p=0.4))
        layers.append(nn.Linear(ncs * out * out, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# helper functions
def denorm(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    ndim = tensor.ndimension()

    if ndim == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    denormalized_tensor = tensor * std + mean
    denormalized_tensor = denormalized_tensor.clamp(0, 1)

    if ndim == 3:
        denormalized_tensor = denormalized_tensor.permute(1, 2, 0)

    return denormalized_tensor


def save_fake_images(index, dir, G, sample_vectors, show=False):
    batch_size = sample_vectors.size(0)
    if batch_size > 16:
        sample_vectors = sample_vectors[:16]
    fake_images = G(sample_vectors)
    if index == -1:
        fake_fname = "fake_images_final.png"
    elif index == -2:
        fake_fname = "fake_images_generator_warmup.png"
    else:
        fake_fname = "fake_images-{0:0=4d}.png".format(index)
    print("Saving", fake_fname)
    save_image(denorm(fake_images.cpu()), os.path.join(dir, fake_fname), nrow=4)
    if show:
        img = Image.open(os.path.join(dir, fake_fname))
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


def sample_image(G, latent_size, device):
    sample_vectors = torch.randn(1, latent_size).to(device)
    fake_images = G(sample_vectors)
    return (
        make_grid(denorm(fake_images.cpu()), nrow=1).permute(1, 2, 0).detach().numpy()
    )
