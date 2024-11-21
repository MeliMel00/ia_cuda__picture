from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data
from PIL import Image
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, H, W = x.size()
        proj_query = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)  # [B, HW, C//8]
        proj_key = self.key(x).view(batch, -1, H * W)  # [B, C//8, HW]
        attention = torch.bmm(proj_query, proj_key)  # [B, HW, HW]
        attention = torch.softmax(attention, dim=-1)
        proj_value = self.value(x).view(batch, -1, H * W)  # [B, C, HW]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(batch, C, H, W)
        return self.gamma * out + x



# Dataset personnalisé pour CelebA-HQ
class CelebAHQDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Chemin vers le dossier contenant les images
        :param transform: Transformations à appliquer aux images
        """
        self.root_dir = root_dir
        self.transform = transform
        # Liste tous les fichiers dans le dossier
        self.image_names = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Charge une image par son nom
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = Image.open(img_name).convert('RGB')  # S'assure que l'image est en RGB

        if self.transform:
            image = self.transform(image)

        return image


# Parser les arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='folder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if torch.backends.mps.is_available() and not opt.mps:
    print("WARNING: You have mps device, to enable macOS GPU run with --mps")
  
if opt.dataroot is None:
    raise ValueError("`dataroot` parameter is required")

# Charger les images avec un Dataset personnalisé
transform = transforms.Compose([
    transforms.Resize(opt.imageSize),
    transforms.CenterCrop(opt.imageSize),
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = CelebAHQDataset(root_dir=opt.dataroot, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

use_mps = opt.mps and torch.backends.mps.is_available()
if opt.cuda:
    device = torch.device("cuda:0")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# Initialisation des poids
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


# Définition du générateur
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.Dropout(0.3),
            SelfAttention(ngf * 8),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


# Définition du discriminateur
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(ndf * 2),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# Configuration des optimizers# Si vous avez d'autres imports ou configurations, conservez-les en dehors de ce bloc

if __name__ == '__main__':  # Ajout du bloc pour éviter les erreurs sur Windows
    # Initialisation des modèles, des optimisateurs et du critère de perte
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # Enregistrer des images pendant l'entraînement
    def save_fake_images(epoch, batches_done):
        # Sauvegarder un lot d'images générées
        with torch.no_grad():
            fake_images = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake_images, os.path.join(opt.outf, f'fake_samples_epoch_{epoch:03d}.png'), normalize=True)

    # Entraînement du modèle
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            # Mettre à jour le discriminateur
            netD.zero_grad()
            
            # Vrai batch d'images
            real_images = data.to(device)
            batch_size = real_images.size(0)
            label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)

            
            # Calcul de la perte du discriminateur pour les vraies images
            output = netD(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            
            # Générer un lot d'images fausses
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label)
            label = label.to(torch.float)
            
            # Calcul de la perte du discriminateur pour les images fausses
            output = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            # Mise à jour du discriminateur
            errD = errD_real + errD_fake
            optimizerD.step()

            # Mettre à jour le générateur
            netG.zero_grad()
            label.fill_(real_label)  # On veut que le générateur trompe le discriminateur
            output = netD(fake_images).view(-1)
            
            # Calcul de la perte du générateur
            errG = criterion(output, label)
            errG.backward()
            
            # Mise à jour du générateur
            optimizerG.step()

            # Afficher les statistiques
            if i % 50 == 0:
                print(f'[{epoch}/{opt.niter}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'[{i * len(real_images)}/{len(dataloader.dataset)}]')

        # Sauvegarder les images générées à chaque époque
        save_fake_images(epoch, i)

        # Sauvegarder les modèles tous les 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(netG.state_dict(), os.path.join(opt.outf, f'netG_epoch_{epoch + 1}.pth'))
            torch.save(netD.state_dict(), os.path.join(opt.outf, f'netD_epoch_{epoch + 1}.pth'))

    # Sauvegarder le dernier modèle
    torch.save(netG.state_dict(), os.path.join(opt.outf, 'netG_final.pth'))
    torch.save(netD.state_dict(), os.path.join(opt.outf, 'netD_final.pth'))

