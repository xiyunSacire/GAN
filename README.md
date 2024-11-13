# 生成对抗网络
##  原理讲解 ：
### 详细原理:
- GAN是一种深度学习模型，最早由Ian Goodfellow及其同事在2014年提出。由生成器（Generator）和判别器（Discriminator）组成。这两个神经网络相互对抗，通过不断调整参数提高自身表现，生成模型尝试生成更逼真的数据，判别模型提高判断数据真实性和生成数据的能力。
- 论文链接：https://arxiv.org/pdf/1406.2661
  
<img width="793" alt="1" src="https://github.com/user-attachments/assets/82eea26b-4666-49e9-a858-3d269845474c">

- GAN结构如上图所示，由一个生成器和判别器构成，将一个随机噪声（随机矩阵）输入给生成器，它将输出一些生成的图片，这些图片会和一些真实的样本数据一起输入给判别器，由判别器判断这些图片的真假，最后返回给生成器一些数据来更新生成器，以此生成更加真实的图片。
### 注意事项：
- **判别器和生成器的更新速度需要差不多**：如果生成器更新速度过快导致判别器无法判断真伪。同样的如果判别器更新速度过快会导致生成器还没有进步就全部被识别出来。所以需要**生成器和判别器的更新速度差不多才能更好的使得它们不断迭代之后收敛**。
# Discriminator：
<img width="360" alt="2" src="https://github.com/user-attachments/assets/ec696248-8035-41c6-bee8-5f39fdf50831">

# Generator：
<img width="233" alt="3" src="https://github.com/user-attachments/assets/4863ceef-7a75-4c35-a0a8-d8f0f466bb33">

## 对于生成器的问题：
早期的时候 G 比较弱，D可以很容易把 G 和 真实数据分开，其中log(1 − D(G(z)))会变成零，导致生成器无法进行迭代。这个问题有许多解决办法，在本次实践内容中**通过交替训练生成器和判别器，并采用适当的损失函数和优化策略，来逐步提升生成器生成逼真图片的能力，同时保持判别器的区分能力**，解决早期生成器较弱的问题。

# 项目实践
## 第一步：引用pytorch库和各项参数定义
<img width="400" alt="4" src="https://github.com/user-attachments/assets/fe5c5939-f3d7-4abc-a0fc-b31369cd561d">
<img width="700" height="400" alt="5" src="https://github.com/user-attachments/assets/a03e9d25-f877-4c69-93c5-203506e59e79">

## 第二步：实现损失函数

### 损失函数的实现：使用 pyTorch 中的 BCELoss 函数
<img width="375" alt="6" src="https://github.com/user-attachments/assets/01ea1a71-c52e-4c10-8354-5cc1b69f9e8a">


### BCELoss：
<img width="681" alt="7" src="https://github.com/user-attachments/assets/ebcd6a08-e6bb-4bd6-8f52-bba168c18b46">

## 第三步：实现生成器和判别器的定义
<img width="469" alt="8" src="https://github.com/user-attachments/assets/e560dad6-1c9b-4ab9-9177-a4d553f7a9c2">
<img width="437" alt="9" src="https://github.com/user-attachments/assets/20fdecbb-b345-430d-bfa9-1460b452d398">

## 第四步：下载实验数据。
<img width="600" alt="10" src="https://github.com/user-attachments/assets/ea47f3be-3f2a-4f3b-8f00-5dc967dcdfbe">

```python

# Configue data loader 配置数据加载器

output_dir = "images"

os.makedirs(output_dir, exist_ok=True)

dataloader = torch.utils.data.DataLoader(

datasets.MNIST(

"./data/mnist", # 数据集存放的路径

train=True, # 指定下载训练集

download=True, # 如果本地没有数据集，下载数据

transform=transforms.Compose([

transforms.Resize(opt.img_size),

transforms.ToTensor(),

transforms.Normalize([0.5], [0.5])

])

),

batch_size=opt.batch_size,

shuffle=True,

)

```

## 第五步：使用优化器

```python

# Optimizers 优化器

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

optimizer D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

```

## 第六步：训练

```python

#-------------#

# training 训练

#-------------#

for epoch in range(opt.n_epochs):

for i, (imgs, _) in enumerate(dataloader):

# Adversarial ground truths 对抗样本和相应真实标签

# valid: 对抗样本的真实标签（1.0），表示图像为真实样本

# fake: 对抗样本的标签（0.0），表示图像为生成的假样本

valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)

fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

# Configure input 配置输入

real_imgs = Variable(imgs.type(Tensor))

```

### 训练生成器

```python

#-------------------------#

# Train Generator 训练生成器

#----------- - -- -------#

optimizer_G.zero_grad()

# Sample noise as generator input 将噪声样本作为生成器输入

z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

# Generate a batch of images 生成一批图像

gen_imgs = generator(z)

# Loss measures generator's ability to fool the discriminator 损失衡量生成器欺骗鉴别器的能力

g_loss = adversarial_loss(discriminator(gen_imgs), valid)

g_loss.backward()

optimizer_G.step()

```

### 生成判别器

```python

# Train Discriminator 训练判别器

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

```

### 打印生成信息

```python

175 print("[Epoch %d/%d]"%(opt.n epochs, opt.n epochs)

176 print(f"Training Completed, The images are located at the following path:fos.path.abspath(output_dir)}")
```
# 原理讲解
## 一、对抗过程
### 生成对抗网络（GAN）的生成器和判别器各自都有一个独立的损失函数，对抗过程即生成器和判别器的损失函数之间的对抗

1.生成器损失函数：
```
g_loss = adversarial_loss(discriminator(gen_imgs),valid

adversarial_loss = torch.nn.BCELoss()
```

生成器损失函数是通过判别器的输出来进行计算的

这里 valid 是一个全为 1 的标签，表示生成器希望判别器将生成的图像判为真实。这个损失函数就是标准的 BCE 损失函数

BCE损失函数本质上是一个二元交叉熵的公式

```

class BCELoss:

def __call__(self, y_pred, y_true):

#确保预测值在(0，1)之间

y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7) # 防止对数函数中的 NaN

loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

return torch.mean(loss)

```

def __call__(self, y_pred, y_true)：函数定义

y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)：1e−7=0.0000001，这段代码使用 torch.clamp 函数将 y_pred 定义为一个张量，并且将值限制在 (0, 1) 的范围内。这样做是为了防止在后续计算对数时出现无效值。

loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))：torch.log: 计算自然对数

这个公式即如果模型预测的合理，那么这个对数的值会接近0，取负号就会很小。如果模型预测的不合理，这个对数的值就会接近负无穷，取了负号之后就会变的很大

### 所以此生成器是以最小化该损失值为目标来进行训练。
#
2.判别器损失函数：

```

d_loss = (real_loss + fake_loss) / 2

```

其中：

```

real_loss = adversarial_loss(discriminator(real_imgs),valid)

```

真实样本的损失（real_loss）：使用 BCE 损失，目标是让判别器将真实图像分类为真实，将真实图像输入判别器并与 valid（全为1）进行比较

```

real_loss = adversarial_loss(discriminator(gen_imgs.detach()),fake)

```

生成样本的损失（fake_loss）：使用 BCE 损失，目标是让判别器将生成的图像分类为假，将生成图像输入判别器并与 fake（全为0）进行比较。这个损失值其实和之前生成器的损失一模一样，但是使用了.detach() 函数来确保生成器的梯度不会影响判别器的更新。因为之前说过，不能同时更新生成器和判别器的参数，否则会导致训练动态失衡和不稳定，影响模型的收敛性。

### 判别器的最终损失是这两个损失的平均值

### 言而简之，GAN整个模型的训练过程是围绕着生成器和判别器损失函数对抗展开的，通过交替更新这两个函数参数，形成一种动态博弈的关系。这种对抗训练的设计是GAN成功的关键因素之一，确保了生成器和判别器在训练中能够不断优化彼此的性能。

***

### 损失函数作用总结：

### 损失函数是生成对抗网络其核心组成部分，对模型的训练和生成结果有着至关重要的影响。

一、合理的损失函数能够确保生成器和判别器之间的平衡。如果其中一个模型过强，另一个模型可能会失败。损失函数的设计可以帮助保持这种平衡，防止一个模型主导训练过程。

二、GAN的训练过程复杂且不稳定，选择合适的损失函数可以提高模型的收敛性。特定的损失函数可以减小训练过程中的震荡，使得模型更快收敛到一个良好的状态。损失函数直接影响生成样本的质量。如果损失函数未能准确反映真实数据分布的复杂性，生成的样本可能会出现模糊、不真实或缺乏多样性等问题。优化损失函数能够改善生成样本的质量。

三、一个好的损失函数能够促使生成器产生多样化的样本，而不是简单地复制训练数据。损失函数的设计应鼓励生成器探索数据分布的不同区域，以提高生成样本的多样性。

不同应用领域可能需要不同的损失函数。例如，图像生成与文本生成的损失函数设计可能大相径庭。针对特定任务的损失函数能够更好地适应具体需求，从而提高模型的性能。

***

# 二、生成器原理：

结构定义部分 该生成器为神经网络结构

```

class Generator(nn.Module):

def _init_(self):

super(Generator, self)._init_()

self.model = nn.Sequential(

*self.block(opt.latent dim, 128, normalize=False),

*self.block(128，256),

*self.block(256，512),

*self.block(512， 1024),

nn.Linear(1024, int(np.prod(img_shape)))，

nn.Tanh()

)

def block(self, in_feat, out_feat, normalize=True) :

layers = [nn.Linear(in _feat, out_feat)]

if normalize:

layers.append(nn.BatchNorm1d(out_feat, 0.8))

layers.append(nn.LeakyReLU(0.2, inplace=True))

return layers

def forward(self, z)：

img = self.model(z)

img = img.view(img.size(0), *img_shape)

return img

```

继承：nn.Module是 PyTorch 中的一个核心类，它是所有神经网络模块的基类。

定义：首先初始化父类nn.Module，定义属性self.model用来存储nn.Sequential容器，接下来进行block的参数设置，把特征维度拓展到1024，然后使用nn.Linear ，将前一个层的输出（具有1024个特征的向量）转换成一个更高维度的向量，其维度与目标图像的总像素数相匹配(“1024”定义了输入数据的特征维度。“int(np.prod(img_shape)))”是图像的像素值) 。最后使用nn.Tanh()激活函数，它将全连接层的输出值限制在 -1 到 1 之间，用于生成图像数据，因为像素值的范围通常在这个区间内。

逐层升维使生成器能够有效地从简单的随机输入构建出复杂而真实的图像，增强了生成能力和多样性。

方法： def block(self, in_feat, out_feat, normalize=True)

（其中，in_feat 是输入特征的维度。out_feat 是输出特征的维度。normalize 是一个布尔值，默认为 True，表示是否在块中包含批量归一化层。这个方法返回一个全连接层layers。）它在 Generator 类中用来构建一个神经网络的子模块，它由一个全连接层、一个可选的批量归一化层和一个 LeakyReLU 激活函数组成。全连接线性层为 layers，nn.BatchNorm1d 是一维批量归一化层。0.8 是批量归一化层的 momentum 参数，它用于计算运行均值和方差。LeakyReLU 是一种改进的 ReLU 激活函数。

言而简之，block 方法的作用是构建生成器中的一组层，逐步将输入特征从一个维度映射到更高的维度。它通过线性变换、可选的批归一化和激活函数（Leaky ReLU）来增强模型的学习能力，帮助生成器逐步学习和组合特征，从而生成更复杂和真实的图像。

方法：def forward(self, z)： 其中 z 是输入到生成器的噪声向量。

forward 方法定义了模型在进行前向传播时的具体操作。在 PyTorch 中，forward 方法是 nn.Module 的子类必须实现的一个方法，它指定了输入数据通过网络时的计算流程。将输入的噪声向量 z 传递给 self.model，self.model 按照顺序执行这些层的计算，生成输出 img。输出 img 是一个一维张量，使用view 方法用于改变张量的形状而不改变其数据。这里，img.size(0) 保持了批次大小不变，*img_shape 将一维张量重塑为具有形状 img_shape 的多维张量，通常是一个三维形状，如 (channels, height, width)，代表图像的通道数、高度和宽度。最后，返回重塑后的图像张量 img，这个张量现在具有与目标图像相同的形状，并且可以用于后续的评估或作为其他模型的输入。

### 总之，forward 方法的作用是接收一个噪声向量，通过神经网络模型将其转换成具有特定形状的图像张量，并返回这个张量。这个过程是生成对抗网络中生成器的核心功能，它负责生成看起来真实的图像。

***

### 模型结构总结

生成器是一个神经网络模型，它的任务是将从潜在空间中随机采样的低维噪声向量转换成高维的、逼真的图像数据。

生成器的网络结构由一系列全连接层组成，这些层被组织在 nn.Sequential 容器中，以确保数据能够顺利地流经每一层。

每个全连接层后都跟着一个可选的批量归一化层和一个 LeakyReLU 激活函数，这样的组合有助于网络学习复杂的数据表示，并防止神经元死亡问题。

网络的最后一部分是一个全连接层，它将1024维的特征向量映射到与目标图像像素数相匹配的维度，然后通过 Tanh 激活函数将输出值限制在 -1 到 1 之间，模拟图像像素值的范围。

当生成器接收到一个随机噪声向量作为输入时，它通过这个复杂的网络结构进行前向传播，最终输出一个与目标图像具有相同维度的假图像。这个假图像可以用于训练判别器，或者在生成对抗网络中进行评估，以生成高质量的图像数据。

一句话概括，生成器的工作就只有接收随机噪声作为输入，并通过神经网络生成看起来像真实数据的图像。

***

### 功能实现部分

生成器实例

```

generator = Generator()

```

使用优化器

```

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

```

其中 torch.optim.Adam 是 PyTorch 库中定义的 Adam 优化器的类，.parameters() 提供给优化器 torch.optim.Adam 一个包含生成器所有需要更新的参数（权重和偏置）的迭代器。

第二个参数是学习率，定义设置默认为 0.0002。学习率是深度学习训练中的一个重要超参数，它决定了每次迭代更新权重时的步长大小。如果学习率太高，可能会导致训练过程中的不稳定和发散；如果学习率太低，训练过程可能会变得非常缓慢。

第三个参数包含两个超参数 beta1 和 beta2，它们控制着梯度及其平方的指数衰减平均。这两个值通常在 0 和 1 之间，并且它们影响着优化器的动量和 RMSProp（一种优化算法，用于训练深度学习模型） 的调整。opt.b1 和 opt.b2 分别是这两个超参数的值，默认分别为 0.5 和 0.999。beta1 影响着一阶矩估计（即梯度的指数衰减平均），而 beta2 影响着二阶矩估计（即梯度平方的指数衰减平均）。这两个参数的值决定了优化器如何利用过去的梯度信息来调整学习率。

### 生成器使用优化器的作用是通过调整其参数来提高生成样本的质量，使生成的图像更具真实性，从而更好地欺骗判别器。优化器帮助生成器在训练过程中有效地学习和优化目标函数，以实现更好的生成效果。

```

# Sample noise as generator input 将噪声样本作为生成器输入

z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent dim))))

```

使用 np.random.normal 函数从标准正态分布（均值为0，标准差为1）中随机生成噪声。这里生成的噪声形状为 (imgs.shape[0], opt.latent_dim)，其中 imgs.shape[0] 是当前批次的大小（batch size），而 opt.latent_dim 是潜在空间的维度，即生成器输入噪声的维度。

Tensor 方法将生成的 NumPy 数组转换为 PyTorch 张量，以便可以在模型中使用。

Variable 方法将 PyTorch 张量封装为一个变量，使其可以进行自动梯度计算。

### 生成的噪声 z 被用作生成器的输入，生成器随后会尝试生成与真实数据相似的假图像。这个过程是 GAN 训练的核心，生成器不断学习如何生成更真实的图像

## 三、判别器原理：
<img width="431" alt="图片1" src="https://github.com/user-attachments/assets/c18fd4c4-0577-432e-86c0-18cfefbc11ad">

- **判别器是一个神经网络模型，它的任务是评估输入图像的真实性，判断其是来自真实数据还是生成器产生的样本。**
     **判别器的网络结构由一系列全连接层组成**，这些层被组织在 nn.Sequential 容器中，以确保数据能够顺利地流经每一层。
     每个全连接层后都跟着一个 LeakyReLU 激活函数，增强了网络的非线性表达能力，并防止神经元死亡问题。
     网络的最后一部分是一个全连接层，将特征映射到一个输出节点，并通过 Sigmoid 激活函数将输出值限制在 0 到 1 之间，表示图像的真实性概率。
     **当判别器接收到一张图像作为输入时，它通过这个网络结构进行前向传播，最终输出一个代表该图像真实性的概率值**。这个概率值可以用于训练生成器和判别器，使得生成对抗网络的训练更为有效。
      **一句话概括，判别器的工作就是接收真实图像和生成器图像作为输入，并通过神经网络判断该图像是否真实**。
## 功能实现部分：
#### 使用优化器

- 其中 torch.optim.Adam 是 PyTorch 库中定义的 Adam 优化器的类，用于优化判别器的参数。.parameters() 方法为优化器提供了一个迭代器，该迭代器包含了判别器模型中所有需要更新的参数（即权重和偏置）。
- 第二个参数是**学习率 lr**，由 opt.lr 指定，其默认值为 0.0002。**学习率是深度学习训练中的关键超参数**，决定了**在每次迭代中权重更新的步长大小**。如果学习率设置得过高，可能导致训练过程不稳定，甚至发生发散；相反，若学习率过低，则训练过程可能会非常缓慢。
- 第三个参数是 **betas**，包含**两个超参数 beta1 和 beta2**，用于控制梯度及其平方的指数衰减平均。这两个值通常在 0 和 1 之间，影响优化器的动量以及对梯度信息的调整。opt.b1 和 opt.b2 分别对应这两个超参数的值，默认设置为 0.5 和 0.999。**beta1 主要影响一阶矩估计（即梯度的指数衰减平均）**，而 **beta2 则影响二阶矩估计（即梯度平方的指数衰减平均）**。这两个参数的选择直接关系到优化器**如何使用历史梯度信息来动态调整学习率**，从而提高训练效果和收敛速度。
- **判别器使用优化器的作用是通过更新其参数来提高对真实和生成样本的辨别能力，从而准确地区分真实数据和生成数据。优化器确保判别器在训练中不断调整以适应生成器的变化，从而增强其分类性能。**
## 四、训练过程
- 首先，**这两个循环构成了训练 GAN 的基本迭代过程**。外层循环中 n_epochs 是一个通过命令行参数传入的变量，**表示训练过程中的总轮数（Epochs）**，即整个数据集将被遍历多少次。
<img width="842" alt="11" src="https://github.com/user-attachments/assets/fb3f7734-becd-4f21-8847-e928e01540fb">
<img width="657" alt="12" src="https://github.com/user-attachments/assets/5ff61b35-7fdd-4337-9a09-12958d557f8b">

### MNIST 
数据集大概有60,000个样本,那么在每个Epoch中，将会有：
<img width="261" alt="13" src="https://github.com/user-attachments/assets/c0c57bdf-a0c1-4a4a-a0b2-dc7bca1657d8">

- 内层循环中，dataloader 是一个**数据加载器**，用于按批次加载训练数据。它可以**从数据集中提取样本并进行批处理**。imgs 代表当前批次的图像数据，每个批次包含 opt.batch_size 个图像。**这个循环表示每轮中进行多少个批次，迭代次数等于批次数量**。具体计算方法在上方。
- 这个循环确保了模型可以**逐批次地处理数据**，而不是一次性处理整个数据集，这对于**内存管理和梯度更新的稳定性都是有益的**。
**Valid 是真实样本的标签，将全部样本标记为 1** ：
imgs.size(0)获取当前批次的图像数量（即样本数量）。Tensor(imgs.size(0), 1)创建一个大小为 (batch_size, 1) 的张量。**fill_(1.0)将该张量中的所有元素填充为 1.0，表示真实样本的标签**。
Variable(..., requires_grad=False)将张量包装成一个变量，并设置 requires_grad=False，表示不需要计算该变量的梯度，因为真实标签在反向传播中不需要更新。
**Fake 是生成样本的标签，将全部样本标记为 0** ：
同样的逻辑，fill_(0.0)将该张量中的所有元素填充为 0.0，表示生成的假样本的标签。
通过 Variable 进行包装，requires_grad=False 表示该变量不需要计算梯度。
**这两者在训练过程中将用于计算损失，帮助判别器（Discriminator）学习区分真实样本和生成的样本**。
real_imgs = Variable(imgs.type(Tensor))  这段是将作为输入的**真实样本**转化为Tensor张量
### 生成器训练过程：
<img width="655" alt="14" src="https://github.com/user-attachments/assets/1526d63d-f055-4f04-8272-6e7d5bd5df21">

首先，每个批次开始之前使用optimizer_G.zero_grad()清零生成器的梯度，以避免梯度累积导致梯度爆炸。
     接着，**创建 z 噪声张量作为生成器的输入**，使用生成器生产一批图像gen_imgs 
     然后，**使用BCE损失函数来计算生成器的损失**，第一个参数是判别器对生成器生成图像的输出（0~1的概率值），第二个参数是真实样本的标签（所有值是1的张量），**这里的g_loss越小，则生成器功能性实现的越好**。
生成器希望判别器将生成的图像误判为真实图像，**因此这里的 valid 张量填充的是 1** ，函数图像如图所示。

<img width="435" alt="44" src="https://github.com/user-attachments/assets/77a5dadd-9d60-416a-ad2d-a308438e12bf">

     当判别器的预测值 y^​ 接近 1 时，**表示判别器认为输入图片是真实的**，此时生成器的损失 L 较小，说明生成器生成的图片成功“欺骗”了判别器，这是我们希望看到的结果。
     当判别器的预测值 y^​ 接近 0 时，**表示判别器认为输入图片是假的**，此时生成器的损失 L 会非常大，这表明生成器生成的图片不够逼真，判别器能够轻易识别出它们是假的。
     **所以生成器在训练过程中的目标就是让这里的g_loss越小越好**。
     最后两步分别是利用之前定义的方法 backward ，计算生成器的梯度，以便在优化步骤中更新生成器的参数。
     使用计算得到的梯度，**调用优化器 optimizer_G 更新生成器的权重，以降低损失**。
### 判别器训练过程：
<img width="929" alt="图片6" src="https://github.com/user-attachments/assets/aabae8d3-5dd9-49c9-9e2b-432dfb93d963">

同样的，训练判别器前先清空其梯度，然后**计算判别器在判别真实图像时的损失 real_loss** ，这个损失同样使用BCE损失函数得到。**接着计算判别器在判别生成图像时的损失 fake_loss** 。**最后判别器的损失函数是取两个损失函数的平均值，这样是因为判别器需要同时学习区分真实图像和生成图像**。通过取平均值，可以**确保判别器在两个任务上都得到平衡的训练，避免过分偏向于其中一个任务**。在训练的早期阶段，生成器可能生成质量较差的图像，这时如果只考虑 fake_loss，判别器可能会过于容易地识别出假图像，导致训练不稳定。通过结合 real_loss，可以让判别器在区分真实图像上也保持一定的压力，从而**提高训练的稳定性**。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTc1NTQwMjAwMywtMTA4MDU5MzU2OSwtMT
A4MDU5MzU2OV19
-->
