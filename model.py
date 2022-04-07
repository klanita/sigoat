import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

class StyleNetworkImage(nn.Module):
    def __init__(self, inplanes=1, planes=16, 
                kernel_size=4, stride=2, padding=1,
                normalization='instance'):
        
        super(StyleNetworkImage, self).__init__()
        
        if normalization == 'instance':
            norm_layer=nn.InstanceNorm2d
        else:
            norm_layer=nn.BatchNorm2d

        self.latent_dim = 1024*2*2            
        output_padding = (0, 0)
        
        encoder = []        
        x = inplanes
        y = planes
        for i in range(7):
            encoder += [
                nn.Conv2d(x, y, kernel_size, stride, padding),
                norm_layer(y),
                nn.LeakyReLU(0.2)]
            x = y
            y = y*2            
        encoder += [nn.Flatten()]
        self.encoder = nn.Sequential(*encoder)
        
        x = 1024
        y = 512
        output_padding = (0, 0) 
        decoder = [nn.Unflatten(1, (x, 2, 2))]  
        decoder_style = [nn.Unflatten(1, (x, 2, 2))]
        for i in range(7):            
            decoder += [
                nn.ConvTranspose2d(x, y, kernel_size, stride, padding,\
                    output_padding, bias=False),
                norm_layer(y),
                nn.LeakyReLU(0.2)]
            decoder_style += [
                nn.ConvTranspose2d(x, y, kernel_size, stride, padding,\
                    output_padding, bias=False),
                norm_layer(y),
                nn.LeakyReLU(0.2)]
#             if i == 3:
#                 output_padding = (0, 0)
            x = y
            y = y//2
        
        decoder += [nn.Conv2d(8, 1, 3, 1, 1)]
        self.decoder = nn.Sequential(*decoder)
        decoder_style += [nn.Conv2d(8, 1, 3, 1, 1)]
        self.decoder_style = nn.Sequential(*decoder_style)
        
        self.mu = nn.Embedding(1, self.latent_dim)
        self.logvar = nn.Embedding(1, self.latent_dim)
        self.logvar.weight.data[0] = torch.zeros(
            [1, self.latent_dim], requires_grad=False)

    def forward(self, imgs, real=False):
        z_s = self.encode(imgs)
        return self.decode(z_s, real=real)

    def encode(self, imgs):
        return self.encoder(imgs)

    def encode_with_style(self, imgs, real=False):
        if real:
            return self.encoder(imgs) + self.sample(z_s.shape)
        else:
            return self.encoder(imgs)

    def sample(self, s):
        std = torch.exp(self.logvar.weight[0].expand(s))
        eps = torch.randn_like(std)
        return eps * std + self.mu.weight[0].expand(s)

    def decode(self, z_s, real=False):
        if real:
            return self.decoder_style(z_s + self.sample(z_s.shape))
        else:
            return self.decoder(z_s)


class unet_block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 inner_block=None,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 output_padding=(0, 0),
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.InstanceNorm2d):
        
        super().__init__()
        self.outermost = outermost
        
        if inner_block is None:
            down_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(1, 0)),
                norm_layer(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            down_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                norm_layer(out_channels),
                nn.LeakyReLU(0.2)
            )

        if outermost:
            in_channels = 16
            up_block = nn.Sequential(
                nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride, padding,\
                output_padding, bias=False),
                norm_layer(in_channels),
                nn.LeakyReLU(0.2))
        else:
            up_block = nn.Sequential(
                nn.ConvTranspose2d(out_channels, int(in_channels), kernel_size, stride, padding,\
                output_padding, bias=False),
                norm_layer(in_channels),
                nn.LeakyReLU(0.2))
        
        blocks_list = [down_block]
        
        if not (inner_block is None):
            blocks_list += [inner_block]
        blocks_list += [up_block]
        
        if outermost:
            blocks_list += [nn.Conv2d(16, 1, 3, 1, 1)]
        
        self.model = nn.Sequential(*blocks_list)
    
    def forward(self, x):
        return self.model(x)

class UNet(nn.Module):
    def __init__(self, norm_layer=nn.InstanceNorm2d):   
        
        super(UNet, self).__init__()
        
        l1 = unet_block(256, 512, norm_layer=norm_layer, output_padding=(1, 0),
            inner_block=None, innermost=True, outermost=False) #inner block
        
        l2 = unet_block(128, 256, norm_layer=norm_layer, output_padding=(1, 0),
            inner_block=l1, innermost=False, outermost=False)
        
        l3 = unet_block(64, 128, norm_layer=norm_layer, output_padding=(1, 0),
            inner_block=l2, innermost=False, outermost=False)
        
        l4 = unet_block(32, 64, norm_layer=norm_layer, output_padding=(1, 0),
            inner_block=l3, innermost=False, outermost=False)
        
        l5 = unet_block(16, 32, norm_layer=norm_layer, output_padding=(0, 0),
            inner_block=l4, innermost=False, outermost=False)
        
        self.model = unet_block(1, 16, norm_layer=norm_layer, output_padding=(0, 0),
            inner_block=l5, innermost=False, outermost=True)
    
    def forward(self, img):
        return self.model(img)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 7168

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

class StyleNetwork(nn.Module):
    def __init__(
        self, img_size, inplanes=1, planes=16, 
        kernel_size=4, stride=2, padding=1,
        normalization='instance'
        ):

        super(StyleNetwork, self).__init__()
        if normalization == 'instance':
            norm_layer=nn.InstanceNorm2d
        elif normalization == 'bacth':
            norm_layer=nn.BatchNorm2d
        else:
            raise NotImplementedError

        if (img_size == 'linear') or (img_size == -1):
            # means we run on the whole image
            latent=14
            output_padding=(1, 0)
            dim2 = 2     
        elif img_size == 'multi':
            latent=14
            output_padding=(1, 0)
            dim2 = 4     
        else:
            # means we run on a crop of the image
            output_padding=(0, 0)
            latent=2
                    
        self.latent_dim = 512*latent*dim2            
        
        encoder = []        
        x = inplanes
        y = planes
        for i in range(6):
            encoder += [
                nn.Conv2d(x, y, kernel_size, stride, padding),
                norm_layer(y),
                nn.LeakyReLU(0.2)]
            if x == 1:
                x = 16
            else:
                x = x*2
            y = y*2
        
        encoder += [nn.Flatten()]
        self.encoder = nn.Sequential(*encoder)
        
        x = 512
        y = 256        
        decoder = [nn.Unflatten(1, (x, latent, dim2))]  
        decoder_style = [nn.Unflatten(1, (x, latent, dim2))]
        for i in range(6):            
            decoder += [
                nn.ConvTranspose2d(x, y, kernel_size, stride, padding,\
                    output_padding, bias=False),
                norm_layer(y),
                nn.LeakyReLU(0.2)]
            decoder_style += [
                nn.ConvTranspose2d(x, y, kernel_size, stride, padding,\
                    output_padding, bias=False),
                norm_layer(y),
                nn.LeakyReLU(0.2)]
            if i == 3:
                output_padding = (0, 0)
            x = x//2
            if y != 16:
                y = y//2
        
        decoder += [nn.Conv2d(16, 1, 3, 1, 1)]
        self.decoder = nn.Sequential(*decoder)
        decoder_style += [nn.Conv2d(16, 1, 3, 1, 1)]
        self.decoder_style = nn.Sequential(*decoder_style)
        
        self.mu = nn.Embedding(1, self.latent_dim)
        self.logvar = nn.Embedding(1, self.latent_dim)
        self.logvar.weight.data[0] = torch.zeros(
            [1, self.latent_dim], requires_grad=False)

    def forward(self, imgs, real=False):
        z_s = self.encode(imgs)
        return self.decode(z_s, real=real)

    def encode(self, imgs):
        return self.encoder(imgs)

    def encode_with_style(self, imgs, real=False):
        if real:
            return self.encoder(imgs) + self.sample(z_s.shape)
        else:
            return self.encoder(imgs)

    def sample(self, s):
        std = torch.exp(self.logvar.weight[0].expand(s))
        eps = torch.randn_like(std)
        return eps * std + self.mu.weight[0].expand(s)

    def decode(self, z_s, real=False):
        if real:
            return self.decoder_style(z_s + self.sample(z_s.shape))
        else:
            return self.decoder(z_s)

class DiscriminatorLatent(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        # TODO change to flexible architecture
        self.linears = nn.Sequential(
            # nn.Linear(latent_dim, n_classes),
            nn.Linear(latent_dim, 1000),

            nn.BatchNorm1d(1000), #remove if needed

            nn.LeakyReLU(0.2),
            nn.Linear(1000, 500),
            nn.LeakyReLU(0.2),

            nn.Linear(500, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.linears(z)

class Discriminator(nn.Module):
    def __init__(self, img_size=-1, inplanes=1, planes=16, kernel_size=4, stride=2, padding=1,
                 norm_layer=nn.InstanceNorm2d, patch=True):
        
        super(Discriminator, self).__init__()
        if img_size == -1:
            # means we run on the whole image
            latent=14
        else:
            latent=2
            
        self.latent_dim = 256*latent
        
        main = []        
        in_channels = inplanes
        out_channels = planes
        for i in range(7):
            main += [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                norm_layer(out_channels),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5)]

            if in_channels == 1:
                in_channels = 16
            else:
                in_channels = in_channels*2
            if out_channels != 512:
                out_channels = out_channels*2
        
        if patch:
            main += [nn.Flatten(), nn.Sigmoid()]
        else:
            main += [nn.Flatten(), nn.Linear(self.latent_dim, 1), nn.Sigmoid()]
        

        self.main = nn.Sequential(*main)
        
    def forward(self, input):
        return self.main(input)

class DiscriminatorSides(nn.Module):
    def __init__(self, inplanes=1, planes=16, kernel_size=4, stride=2, padding=1,
                 norm_layer=nn.InstanceNorm2d):
        
        super(DiscriminatorSides, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
                nn.InstanceNorm2d(16),
                nn.LeakyReLU(0.2),

                nn.Conv2d(16, 32, 4, 2, 1),
                nn.InstanceNorm2d(32),
                nn.LeakyReLU(0.2),

                nn.Conv2d(32, 64, 4, 2, 1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 128, 4, 2, 1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 256, 4, 2, 1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 512, 4, 2, 1),  # 512-14-4
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2),

                nn.Sigmoid())
                
    def forward(self, input):
        return self.main(input)


class SidesReconstruction(nn.Module):
    # def __init__(self, latent_dim=14336, n_label=2):
    def __init__(self, latent_dim=14336, img_size=-1, inplanes=1, planes=16, 
                kernel_size=4, stride=2, padding=1,
                norm_layer=nn.InstanceNorm2d):

        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            # nn.Conv2d(256, 512, 4, 2, (1, 0)),  # 512-14-4
            # nn.InstanceNorm2d(512),
            # nn.LeakyReLU(0.2),

            # nn.Conv2d(512, 512, 4, 2, 1),  # 512-14-4
            # nn.InstanceNorm2d(512),
            # nn.LeakyReLU(0.2),
        )

        self.sides_head_center = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),  # 512-14-4
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),                               
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

        self.sides_head_left = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, (1, 0)),  # 512-14-4
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            # nn.Flatten(),
            # nn.Linear(latent_dim, 7168),
            # nn.LeakyReLU(0.2),

            # nn.Unflatten(1, (512, 14, 1)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1,
                            output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1,
                            output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, 1, 1)
            
        )

        self.sides_head_right = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, (1, 0)),  # 512-14-4
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            # nn.Flatten(),
            # nn.Linear(latent_dim, 7168),
            # nn.LeakyReLU(0.2),
            
            # nn.Unflatten(1, (512, 14, 1)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

        
    def forward(self, imgs):
        z_s = self.encoder(imgs)

        reconsts_center = self.decode(z_s, sides='center').clone().detach()
        reconsts_left = self.decode(z_s, sides='left').clone().detach()
        reconsts_right = self.decode(z_s, sides='right').clone().detach()
        
        reconsts = torch.cat((reconsts_left, reconsts_center, reconsts_right), 3)

        return reconsts

    def encode(self, imgs):
        z_s = self.encoder(imgs)
        return z_s

    def decode(self, z_s, sides='center'):        
        if sides == 'right':            
            reconsts = self.sides_head_right(z_s)

        elif sides == 'left':
            reconsts = self.sides_head_left(z_s)

        elif sides == 'center':
            reconsts = self.sides_head_center(z_s)
        
        else:
            raise NotImplementedError

        return reconsts


class SidesDecoder(nn.Module):
    # def __init__(self, latent_dim=14336, n_label=2):
    def __init__(self, latent_dim=14336, img_size=-1, inplanes=1, planes=16, 
                kernel_size=4, stride=2, padding=1,
                norm_layer=nn.InstanceNorm2d):

        super().__init__()
        self.latent_dim = latent_dim

        self.sides_head_left = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 7168),
            nn.LeakyReLU(0.2),

            nn.Unflatten(1, (512, 14, 1)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1,
                            output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1,
                            output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

        self.sides_head_right = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 7168),
            nn.LeakyReLU(0.2),
            
            nn.Unflatten(1, (512, 14, 1)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

        
    def forward(self, imgs):
        z_s = self.encoder(imgs)

        reconsts_center = self.decode(z_s, sides='center').clone().detach()
        reconsts_left = self.decode(z_s, sides='left').clone().detach()
        reconsts_right = self.decode(z_s, sides='right').clone().detach()
        
        reconsts = torch.cat((reconsts_left, reconsts_center, reconsts_right), 3)

        return reconsts

    def decode(self, z_s, sides):        
        if sides == 'right':            
            reconsts = self.sides_head_right(z_s)

        elif sides == 'left':
            reconsts = self.sides_head_left(z_s)
        
        else:
            raise NotImplementedError

        return reconsts


class FaderNetwork(nn.Module):
    def __init__(
        self,
        latent_dim=14336,
        n_label=2,
        normalization='batch'):

        super().__init__()
        self.latent_dim = latent_dim

        if normalization == 'instance':
            norm_layer=nn.InstanceNorm2d
        else:
            norm_layer=nn.BatchNorm2d

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            norm_layer(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 32, 4, 2, 1),
            norm_layer(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            norm_layer(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            norm_layer(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            norm_layer(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),  # 512-14-4
            norm_layer(512),
            nn.LeakyReLU(0.2),
        )

        self.sides_head_center = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),                               
            norm_layer(256),
            # nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

        self.sides_head_left = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 7168),
            # nn.LeakyReLU(0.2),

            nn.Unflatten(1, (512, 14, 1)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1,
                            output_padding=(1, 0), bias=False),
            norm_layer(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1,
                            output_padding=(1, 0), bias=False),
            norm_layer(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

        self.sides_head_right = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 7168),
            # nn.LeakyReLU(0.2),
            
            nn.Unflatten(1, (512, 14, 1)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

        # self.lt = nn.Embedding(n_label, latent_dim)
        
    def forward(self, imgs):
        z_s = self.encoder(imgs)

        reconsts_center = self.decode(z_s, sides='center')
        reconsts_left = self.decode(z_s, sides='left')
        reconsts_right = self.decode(z_s, sides='right')
        
        reconsts = torch.cat((reconsts_left, reconsts_center, reconsts_right), 3)

        return reconsts

    def encode(self, imgs):
        z_s = self.encoder(imgs)
        return z_s

    def decode(self, z_s, sides='center'):        
        if sides == 'right':            
            reconsts = self.sides_head_right(z_s)

        elif sides == 'left':
            reconsts = self.sides_head_left(z_s)

        elif sides == 'center':
            reconsts = self.sides_head_center(z_s)
        
        else:
            raise NotImplementedError

        return reconsts


class ReconstructionNetwork(nn.Module):
    def __init__(self, inplanes=1, planes=16, 
                kernel_size=4, stride=2, padding=1,
                norm_layer=nn.InstanceNorm2d):
        
        super(ReconstructionNetwork, self).__init__()
        
        latent=14
        output_padding=(1, 0)                        
        self.latent_dim = 512*latent*2            
        
        encoder = []        
        x = inplanes
        y = planes
        for i in range(6):
            encoder += [
                nn.Conv2d(x, y, kernel_size, stride, padding),
                norm_layer(y),
                nn.LeakyReLU(0.2)]
            if x == 1:
                x = 16
            else:
                x = x*2
            y = y*2
        
        # encoder += [nn.Flatten()] + [nn.Linear(self.latent_dim, self.latent_dim*4)]
        self.encoder = nn.Sequential(*encoder)
        
        x = 512
        y = 256
        # decoder = [nn.Unflatten(1, (x, latent, 8))]  
        decoder = []
        
        for i in range(6):            
            decoder += [
                nn.ConvTranspose2d(x, y, kernel_size, stride, padding,\
                    output_padding, bias=False),
                norm_layer(y),
                nn.LeakyReLU(0.2)]
            if i == 3:
                output_padding = (0, 0)
            x = x//2
            if y != 16:
                y = y//2
        
        decoder += [nn.Conv2d(16, 1, 3, 1, 1)]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, imgs):
        z = self.encoder(imgs)
        return self.decoder(z)


class StyleNetworkAblation(nn.Module):
    def __init__(
        self,
        img_size,
        inplanes=1,
        planes=16, 
        kernel_size=4,
        stride=2,
        padding=1,
        normalization='instance',
        output_padding=(1, 0),
        n_layers=6
    ):
        if normalization == 'instance':
            norm_layer=nn.InstanceNorm2d
        else:
            norm_layer=nn.BatchNorm2d        
        
        super(StyleNetworkAblation, self).__init__()
        
        if n_layers == 6:
            latent=14
            if (img_size == 'linear'):
                dim2 = 2     
            elif img_size == 'multi':
                dim2 = 4 
            else:
                raise NotImplementedError
        else:
            latent=7
            if (img_size == 'linear'):
                dim2 = 1
            elif img_size == 'multi':
                dim2 = 2    
            else:
                raise NotImplementedError
            
                    
        self.latent_dim = 512*latent*dim2            
        
        encoder = []        
        x = inplanes
        y = planes
        for i in range(n_layers):
            if i == 6:
                encoder += [
                    nn.Conv2d(x, y, kernel_size, stride, padding),
                    norm_layer(y),
                    nn.LeakyReLU(0.2)]
            else:
                encoder += [
                    nn.Conv2d(x, y, kernel_size, stride, (1, 1)),
                    norm_layer(y),
                    nn.LeakyReLU(0.2)]
            x = y
            y = y*2
        
        encoder += [nn.Flatten()]
        self.encoder = nn.Sequential(*encoder)
        
        if n_layers == 6:
            x = 512
            y = 256   
        else:
            x = 1024
            y = 512        
        decoder = [nn.Unflatten(1, (x, latent, dim2))]  
        decoder_style = [nn.Unflatten(1, (x, latent, dim2))]
        
        for i in range(n_layers):            
            if i == 4:
                output_padding = (0, 0)
                
            decoder += [
                nn.ConvTranspose2d(
                    x, y, kernel_size, stride, padding,\
                    output_padding, bias=False
                ),
                norm_layer(y),
                nn.LeakyReLU(0.2)]
            
            decoder_style += [
                nn.ConvTranspose2d(
                    x, y, kernel_size, stride, padding,\
                    output_padding, bias=False
                ),
                norm_layer(y),
                nn.LeakyReLU(0.2)
            ]
            
            x = x//2
            if y != 16:
                y = y//2
        
        decoder += [nn.Conv2d(16, 1, 3, 1, (1, 1))]
        self.decoder = nn.Sequential(*decoder)
        
        decoder_style += [nn.Conv2d(16, 1, 3, 1, 1)]
        self.decoder_style = nn.Sequential(*decoder_style)
        
        self.mu = nn.Embedding(1, self.latent_dim)
        self.logvar = nn.Embedding(1, self.latent_dim)
        self.logvar.weight.data[0] = torch.zeros(
            [1, self.latent_dim], requires_grad=False)

    def forward(self, imgs, real=False):
        z_s = self.encode(imgs)
        return self.decode(z_s, real=real)

    def encode(self, imgs):
        return self.encoder(imgs)

    def encode_with_style(self, imgs, real=False):
        if real:
            return self.encoder(imgs) + self.sample(z_s.shape)
        else:
            return self.encoder(imgs)

    def sample(self, s):
        std = torch.exp(self.logvar.weight[0].expand(s))
        eps = torch.randn_like(std)
        return eps * std + self.mu.weight[0].expand(s)

    def decode(self, z_s, real=False):
        if real:
            return self.decoder_style(z_s + self.sample(z_s.shape))
        else:
            return self.decoder(z_s)


class FullNetwork(nn.Module):
    def __init__(
        self,
        img_size,
        inplanes=1,
        planes=16, 
        kernel_size=4,
        stride=2,
        padding=1,
        normalization='instance',
        output_padding=(1, 0),
        n_layers=6
    ):
        if normalization == 'instance':
            norm_layer=nn.InstanceNorm2d
        else:
            norm_layer=nn.BatchNorm2d        
        
        super(FullNetwork, self).__init__()
        
        if n_layers == 6:
            latent=14
            if (img_size == 'linear'):
                dim2 = 2     
            elif img_size == 'multi':
                dim2 = 4 
        else:
            latent=7
            if (img_size == 'linear'):
                dim2 = 1
            elif img_size == 'multi':
                dim2 = 2        
                    
        self.latent_dim = 512*latent*dim2            
        
        encoder = []        
        x = inplanes
        y = planes
        for i in range(n_layers):
            if i == 6:
                encoder += [
                    nn.Conv2d(x, y, kernel_size, stride, padding),
                    norm_layer(y),
                    nn.LeakyReLU(0.2)]
            else:
                encoder += [
                    nn.Conv2d(x, y, kernel_size, stride, (1, 1)),
                    norm_layer(y),
                    nn.LeakyReLU(0.2)]
            x = y
            y = y*2
        
        encoder += [nn.Flatten()]
        self.encoder = nn.Sequential(*encoder)
        
        if n_layers == 6:
            x = 512
            y = 256   
        else:
            x = 1024
            y = 512        
        decoder = [nn.Unflatten(1, (x, latent, dim2))]  
        decoder_style = [nn.Unflatten(1, (x, latent, dim2))]
        
        for i in range(n_layers):            
            if i == 4:
                output_padding = (0, 0)
                
            decoder += [
                nn.ConvTranspose2d(
                    x, y, kernel_size, stride, padding,\
                    output_padding, bias=False
                ),
                norm_layer(y),
                nn.LeakyReLU(0.2)]
            
            decoder_style += [
                nn.ConvTranspose2d(
                    x, y, kernel_size, stride, padding,\
                    output_padding, bias=False
                ),
                norm_layer(y),
                nn.LeakyReLU(0.2)
            ]
            
            x = x//2
            if y != 16:
                y = y//2
        
        decoder += [nn.Conv2d(16, 1, 3, 1, (1, 1))]
        self.decoder = nn.Sequential(*decoder)
        
        decoder_style += [nn.Conv2d(16, 1, 3, 1, 1)]
        self.decoder_style = nn.Sequential(*decoder_style)
        
        self.mu = nn.Embedding(1, self.latent_dim)
        self.logvar = nn.Embedding(1, self.latent_dim)
        self.logvar.weight.data[0] = torch.zeros(
            [1, self.latent_dim], requires_grad=False)

        self.sides_head_left = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.latent_dim, 7168),
            nn.LeakyReLU(0.2),

            nn.Unflatten(1, (512, 14, 1)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1,
                            output_padding=(1, 0), bias=False),
            norm_layer(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1,
                            output_padding=(1, 0), bias=False),
            norm_layer(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

        self.sides_head_right = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.latent_dim, 7168),
            nn.LeakyReLU(0.2),
            
            nn.Unflatten(1, (512, 14, 1)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1,
                               output_padding=(1, 0), bias=False),
            norm_layer(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

    def forward(self, imgs, real=False, full=False):
        z_s = self.encode(imgs)
        if full:
            reconsts_center = self.decode(z_s, real=real, sides='center').clone().detach()
            reconsts_left = self.decode(z_s, sides='left').clone().detach()
            reconsts_right = self.decode(z_s, sides='right').clone().detach()
            return torch.cat((reconsts_left, reconsts_center, reconsts_right), 3)
        else:
            return self.decode(z_s, real=real, sides='center')

    def encode(self, imgs):
        return self.encoder(imgs)

    def encode_with_style(self, imgs, real=False):
        if real:
            return self.encoder(imgs) + self.sample(z_s.shape)
        else:
            return self.encoder(imgs)

    def sample(self, s):
        std = torch.exp(self.logvar.weight[0].expand(s))
        eps = torch.randn_like(std)
        return eps * std + self.mu.weight[0].expand(s)

    def decode(self, z_s, real=False, sides='center'):
        if sides == 'center':
            if real:
                # return self.decode(z_s + self.sample(z_s.shape))
                return self.decoder_style(z_s + self.sample(z_s.shape))
            else:
                return self.decoder(z_s)
            
        if sides == 'right':            
            return self.sides_head_right(z_s)

        elif sides == 'left':
            return self.sides_head_left(z_s)
