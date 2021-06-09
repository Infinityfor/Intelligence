"""
通过生成器加载模型，生成图片
"""

#必须保证以下库全部导入
import  os
import torch
import scipy
from scipy import stats
import shutil
import torch.nn as nn
from torchvision.utils import save_image
from torch.nn.utils import spectral_norm
import torch.nn.functional as F


#存储路径
user_images_unzipped_path = 'output_images'
device = 'cpu'

config = {'Generator': {'latent_dim': 120,
                        'embed_dim': 32,
                        'ch': 64,
                        'num_classes': 120,
                        'use_attn': True},
          'sample_latents': {'latent_dim': 120,
                             'num_classes': 120}}

#以下为Generator模型构建
class CBN2d(nn.Module):
    def __init__(self, num_features, num_conditions):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = spectral_norm(nn.Conv2d(num_conditions, num_features * 2, kernel_size=1, bias=False))

        nn.init.orthogonal_(self.embed.weight.data)

    def forward(self, x, y):
        out = self.bn(x)
        embed = self.embed(y.unsqueeze(2).unsqueeze(3))
        gamma, beta = embed.chunk(2, dim=1)
        out = (1.0 + gamma) * out + beta

        return out


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conditions, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.cbn1 = CBN2d(in_channels, num_conditions)
        self.cbn2 = CBN2d(out_channels, num_conditions)
        if self.learnable_sc:
            self.conv_sc = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.relu = nn.ReLU()

        nn.init.orthogonal_(self.conv1.weight.data)
        nn.init.orthogonal_(self.conv2.weight.data)
        if self.learnable_sc:
            nn.init.orthogonal_(self.conv_sc.weight.data)

    def _upsample_conv(self, x, conv):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = conv(x)

        return x

    def _residual(self, x, y):
        x = self.relu(self.cbn1(x, y))
        x = self._upsample_conv(x, self.conv1) if self.upsample else self.conv1(x)
        x = self.relu(self.cbn2(x, y))
        x = self.conv2(x)

        return x

    def _shortcut(self, x):
        if self.learnable_sc:
            x = self._upsample_conv(x, self.conv_sc) if self.upsample else self.conv_sc(x)

        return x

    def forward(self, x, y):
        return self._shortcut(x) + self._residual(x, y)



class Attention(nn.Module):
    def __init__(self, channels, reduction_attn=8, reduction_sc=2):
        super().__init__()
        self.channles_attn = channels // reduction_attn
        self.channels_sc = channels // reduction_sc

        self.conv_query = spectral_norm(nn.Conv2d(channels, self.channles_attn, kernel_size=1, bias=False))
        self.conv_key = spectral_norm(nn.Conv2d(channels, self.channles_attn, kernel_size=1, bias=False))
        self.conv_value = spectral_norm(nn.Conv2d(channels, self.channels_sc, kernel_size=1, bias=False))
        self.conv_attn = spectral_norm(nn.Conv2d(self.channels_sc, channels, kernel_size=1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))

        nn.init.orthogonal_(self.conv_query.weight.data)
        nn.init.orthogonal_(self.conv_key.weight.data)
        nn.init.orthogonal_(self.conv_value.weight.data)
        nn.init.orthogonal_(self.conv_attn.weight.data)

    def forward(self, x):
        batch, _, h, w = x.size()

        proj_query = self.conv_query(x).view(batch, self.channles_attn, -1)
        proj_key = F.max_pool2d(self.conv_key(x), 2).view(batch, self.channles_attn, -1)

        attn = torch.bmm(proj_key.permute(0, 2, 1), proj_query)
        attn = F.softmax(attn, dim=1)

        proj_value = F.max_pool2d(self.conv_value(x), 2).view(batch, self.channels_sc, -1)
        attn = torch.bmm(proj_value, attn)
        attn = attn.view(batch, self.channels_sc, h, w)
        attn = self.conv_attn(attn)

        out = self.gamma * attn + x

        return out


class Generator(nn.Module):
    def __init__(self, latent_dim, ch, num_classes, embed_dim, use_attn=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.ch = ch
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_attn = use_attn
        self.num_chunk = 5
        num_latents = self.__get_num_latents()

        self.embed = nn.Embedding(num_classes, embed_dim)
        self.fc = spectral_norm(nn.Linear(num_latents[0], ch * 8 * 4 * 4, bias=False))
        self.block1 = GBlock(ch * 8, ch * 8, num_latents[1], upsample=True)
        self.block2 = GBlock(ch * 8, ch * 4, num_latents[2], upsample=True)
        self.block3 = GBlock(ch * 4, ch * 2, num_latents[3], upsample=True)
        if use_attn:
            self.attn = Attention(ch * 2)
        self.block4 = GBlock(ch * 2, ch, num_latents[4], upsample=True)
        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU()
        self.conv_last = spectral_norm(nn.Conv2d(ch, 3, kernel_size=3, padding=1, bias=False))
        self.tanh = nn.Tanh()

        nn.init.orthogonal_(self.embed.weight.data)
        nn.init.orthogonal_(self.fc.weight.data)
        nn.init.orthogonal_(self.conv_last.weight.data)
        nn.init.constant_(self.bn.weight.data, 1.0)
        nn.init.constant_(self.bn.bias.data, 0.0)

    def __get_num_latents(self):
        xs = torch.empty(self.latent_dim).chunk(self.num_chunk)
        num_latents = [x.size(0) for x in xs]
        for i in range(1, self.num_chunk):
            num_latents[i] += self.embed_dim

        return num_latents

    def forward(self, x, y):
        xs = x.chunk(self.num_chunk, dim=1)
        y = self.embed(y)

        h = self.fc(xs[0])
        h = h.view(h.size(0), self.ch * 8, 4, 4)
        h = self.block1(h, torch.cat([y, xs[1]], dim=1))
        h = self.block2(h, torch.cat([y, xs[2]], dim=1))
        h = self.block3(h, torch.cat([y, xs[3]], dim=1))
        if self.use_attn:
            h = self.attn(h)
        h = self.block4(h, torch.cat([y, xs[4]], dim=1))
        h = self.relu(self.bn(h))
        out = self.tanh(self.conv_last(h))

        return out


#Generator模型构建完成


#截断参数，用于校准生成图片
def truncated_normal(size, threshold=2.0, dtype=torch.float32, device='cpu'):
    x = scipy.stats.truncnorm.rvs(-threshold, threshold, size=size)
    x = torch.from_numpy(x).to(device, dtype)

    return x

#生成图片
def generate_eval_samples(generator, batch_size, latent_dim, num_classes):
    latents = truncated_normal((batch_size, latent_dim), dtype=torch.float32, device=device)
    labels = torch.randint(0, num_classes, size=(batch_size,), dtype=torch.long, device=device)

    with torch.no_grad():
        imgs = (generator(latents, labels) + 1) / 2

    return imgs


#打包所生成的图片
def make_submissions(generator, user_images_unzipped_path, latent_dim, num_classes):
    if not os.path.exists(user_images_unzipped_path):
        os.mkdir(user_images_unzipped_path)

    sample_batch_size = 50
    num_samples = 100

    for i in range(0, num_samples, sample_batch_size):
        imgs = generate_eval_samples(generator, sample_batch_size, latent_dim, num_classes)
        for j, img in enumerate(imgs):
            save_image(img, os.path.join(user_images_unzipped_path, f'image_{i + j:05d}.png'))

    shutil.make_archive('images', 'zip', user_images_unzipped_path)


#通过加载模型数据，创建生成器netG
netG = Generator(**config['Generator']).to(device, torch.float32)
netG.load_state_dict(torch.load('Generator.pth', map_location='cpu'));#注意文件路径


#结果集生成
make_submissions(netG, user_images_unzipped_path, **config['sample_latents'])#**表示输入多组参数