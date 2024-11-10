# 无条件生成对抗网络（Unconditional Generative Adversarial Network，简称UGAN）中的全连接（Fully Connected, FC） GAN，或称为 MLP GAN（Multi-Layer Perceptron GAN）
import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

# 创建一个参数解析器，用于处理命令行参数
parser = argparse.ArgumentParser()
# 添加训练轮数参数，帮助信息说明此参数用于指定训练的总轮数
# 修改--n_epochs 的 default 更改训练轮数
parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
# 添加批量大小参数，默认为128，帮助信息说明此参数用于指定每次训练时使用的样本数
parser.add_argument("--batch_size", type=int, default=128, help="Number of samples per training batch")
# 添加学习率参数，默认为0.0002，帮助信息说明此参数用于指定Adam优化器的学习率
parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for the Adam optimizer")
# 添加第一个动量系数参数，默认为0.5，帮助信息说明此参数用于指定Adam优化器的第一个动量系数
parser.add_argument("--b1", type=float, default=0.5, help="Beta1 parameter for Adam optimizer")
# 添加第二个动量系数参数，默认为0.999，帮助信息说明此参数用于指定Adam优化器的第二个动量系数
parser.add_argument("--b2", type=float, default=0.999, help="Beta2 parameter for Adam optimizer")
# 添加CPU核心数参数，默认为8，帮助信息说明此参数用于指定用于数据加载的CPU核心数量
parser.add_argument("--n_cpu", type=int, default=8, help="Number of CPU cores for data loading")
# 添加潜在空间维度参数，默认为100，帮助信息说明此参数用于指定生成模型的潜在空间维度
parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of the latent space")
# 添加图像大小参数，默认为28，帮助信息说明此参数用于指定输入图像的尺寸（高度和宽度）
parser.add_argument("--img_size", type=int, default=28, help="Size of the input images (height and width)")
# 添加图像通道数参数，默认为1，帮助信息说明此参数用于指定输入图像的通道数（如灰度或RGB）
parser.add_argument("--channels", type=int, default=1, help="Number of channels in the input images (1 for grayscale, 3 for RGB)")
# 添加样本间隔参数，默认为400，帮助信息说明此参数用于指定每多少步生成一次样本
parser.add_argument("--sample_interval", type=int, default=400, help="Interval between image samples during training")
# 解析命令行参数
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = torch.cuda.is_available()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            *self.block(opt.latent_dim, 128, normalize=False),
            *self.block(128, 256),
            *self.block(256, 512),
            *self.block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def block(self, in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Loss function 损失函数(二进制交叉熵损失对象)
adversarial_loss = torch.nn.BCELoss()

# initialize generator and discriminator 生成判别器和生成器
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configue data loader 配置数据加载器
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",  # 数据集存放的路径
        train=True,      # 指定下载训练集
        download=True,   # 如果本地没有数据集，下载数据
        transform=transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers 优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# d_losses = []  # 判别器损失
# g_losses = []  # 生成器损失

#-------------#
# training 训练
#-------------#
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        
        #Adversarial ground truths 对抗样本和相应真实标签
        # valid: 对抗样本的真实标签（1.0），表示图像为真实样本
        # fake: 对抗样本的标签（0.0），表示图像为生成的假样本
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        
        #Configure input 配置输入
        real_imgs = Variable(imgs.type(Tensor))

        #-------------------------#
        # Train Generator 训练生成器
        #-------------------------#
        optimizer_G.zero_grad()
        
        # Sample noise as generator input 将噪声样本作为生成器输入
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images 生成一批图像
        gen_imgs = generator(z)
        
        # Loss measures generator's ability to fool the discriminator 损失衡量生成器欺骗鉴别器的能力
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        
        g_loss.backward()
        optimizer_G.step()
        
        #-----------------------------#
        # Train Discriminator 训练判别器
        #-----------------------------#
        
        optimizer_D.zero_grad()
                
        # Measure discriminator's ability to classify real from generated samples 衡量鉴别器区分真实样本和生成样本的能力
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        optimizer_D.step()

        print("[Epoch %d/%d][Batch %d/%d] [D loss: %f] [G loss: %f]"
              % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

    # # 绘制损失图像（每个 epoch 结束时）
    # plt.figure(figsize=(10, 5))
    # plt.plot(d_losses, label="Discriminator Loss", color='red')
    # plt.plot(g_losses, label="Generator Loss", color='blue')
    # plt.xlabel("Training Iterations")
    # plt.ylabel("Loss")
    # plt.title(f"Epoch {epoch + 1}/{opt.n_epochs} Losses")
    # plt.legend()
    # # 确保文件夹存在
    # os.makedirs('plots', exist_ok=True)
    # # 保存损失图像到 plots 文件夹
    # plt.savefig(f"plots/losses_epoch_{epoch + 1}.png")
    # plt.close()
    
print("[Epoch %d/%d]" % (opt.n_epochs, opt.n_epochs))
print(f"Training Completed, The images are located at the following path:{os.path.abspath(output_dir)}")