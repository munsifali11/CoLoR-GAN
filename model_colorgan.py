import math
import random
import functools
import operator

import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix


class LoRAFC(nn.Module):
    def __init__(self, in_dim, out_dim, rank, bias=False, lora_alpha_fc = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.bias = bias
        self.lora_alpha_fc  = lora_alpha_fc
        
        if self.rank is not None:
            self.lora_A = nn.Parameter(torch.randn(self.in_dim, self.rank) / math.sqrt(self.rank)) # [d_in, r]
            self.lora_B = nn.Parameter(torch.zeros(self.out_dim, self.rank) / math.sqrt(self.rank)) # [d_out, r]            

            if self.bias:
                self.lora_bias = nn.Parameter(torch.ones(self.out_dim)) # [d_out]
        
    def forward(self, w, b=None):
        if self.rank is not None:
            w_add = self.lora_B @ self.lora_A.transpose(1, 0) # [d_out, d_in]
            w_add = self.lora_alpha_fc * w_add
            w_hat = w + w_add
                
            if self.bias and b is not None:
                b_hat = b + self.lora_bias
            else:
                b_hat = None

        return w_hat, b_hat


class LLoRAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, rank, use_act=True, lora_alpha_conv = 1.0):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.rank = rank
        self.lora_alpha_conv   = lora_alpha_conv
        #self.lora_alpha_conv = nn.Parameter(torch.tensor(lora_alpha_conv))
        if use_act:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        # w: [C_out, C_in, k, k]
        # b: [C_out, k, k]

        if self.rank is not None:
            self.lora_B = nn.Parameter(torch.randn(self.out_channel, self.rank) / math.pow(self.rank, 2/3)) # [C_out, r]
            self.lora_B_inst = nn.Parameter(torch.zeros(self.rank, self.rank, self.kernel_size ** 2) / math.pow(self.rank, 2/3)) # [r, r, k^2]
            self.lora_A = nn.Parameter(torch.randn(self.in_channel, self.rank) / math.pow(self.rank, 2/3)) # [C_in, r]



    def forward(self, w):
        B = self.lora_B @ self.lora_B_inst # [r, C_out, k^2]        
        B = self.activation(B)            
        AB = self.lora_A @ B.transpose(1, 0) # [C_out, C_in, k^2]        
        w_add = self.activation(AB)
        w_add = self.lora_alpha_conv*w_add
        w_hat = w + w_add.view(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size) # [C_out, C_in, k, k]
        
        return w_hat


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class CoLoREqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, rank, use_act, lifelong, lora_alpha_conv,
            stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )

        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.lifelong = lifelong
        self.stride = stride
        self.padding = padding

        if self.lifelong:
            self.weight_modulation = LLoRAConv(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, rank=rank, use_act=use_act, lora_alpha_conv=lora_alpha_conv, bias=bias)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        if self.lifelong:
            weight, bias = self.weight_modulation(self.weight, self.bias)
        else:
            weight, bias = self.weight, self.bias

        out = conv2d_gradfix.conv2d(
            input,
            weight * self.scale,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class CoLoREqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, rank, lifelong, lora_alpha_fc,
            bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.lifelong = lifelong

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        if self.lifelong:
            self.weight_modulation = LoRAFC(in_dim=in_dim, out_dim=out_dim, rank=rank, bias=bias, 
                                            lora_alpha_fc=lora_alpha_fc)

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.lifelong:
            weight, bias = self.weight_modulation(self.weight, self.bias)
        else:
            weight, bias = self.weight, self.bias

        if self.activation:
            out = F.linear(input, weight * self.scale)
            out = fused_leaky_relu(out, bias * self.lr_mul)

        else:
            out = F.linear(
                input, weight * self.scale, bias=bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class CoLoRConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        rank,
        use_act,
        lifelong,
        lora_alpha_conv,
        lora_alpha_fc,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.rank = rank
        self.use_act = use_act
        self.lifelong = lifelong
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = CoLoREqualLinear(style_dim, in_channel, rank=rank, lifelong=lifelong, bias_init=1, 
                                           lora_alpha_fc=lora_alpha_conv)
        if self.lifelong:
            self.weight_modulation = LLoRAConv(in_channel, out_channel, kernel_size, rank=rank, use_act=use_act, lora_alpha_conv=lora_alpha_conv,)

        self.demodulate = demodulate
        self.fused = fused
        self.bias = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        if self.lifelong is True:
            weight = self.weight_modulation(self.weight)
        else:
            weight = self.weight

        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class CoLoRStyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        rank,
        use_act,
        lifelong,
        lora_alpha_conv,
        lora_alpha_fc,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = CoLoRConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            rank,
            use_act,
            lifelong,
            lora_alpha_conv = lora_alpha_conv,
            lora_alpha_fc=lora_alpha_fc,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class CoLoRToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, rank, use_act, lifelong, lora_alpha_conv, lora_alpha_fc,
                 upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = CoLoRConv2d(in_channel, 3, 1, style_dim,
                                            rank, use_act, lifelong, lora_alpha_conv=lora_alpha_conv, lora_alpha_fc =lora_alpha_fc,
                                            demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class CoLoRGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        rank,
        use_act,
        lifelong,
        lora_alpha_conv,
        lora_alpha_fc,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()
        self.lfs = lifelong

        if lifelong is True:
            print("*"*20)
            print("Use CoLoR learning for Generator")

        self.size = size
        self.style_dim = style_dim
        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                CoLoREqualLinear(
                    style_dim, style_dim, rank=rank, lifelong=lifelong, lora_alpha_fc=lora_alpha_fc,
                    lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = CoLoRStyledConv(
            self.channels[4], self.channels[4], 3, style_dim,
            rank=rank, use_act=use_act, lifelong=lifelong, lora_alpha_conv = lora_alpha_conv,
            lora_alpha_fc = lora_alpha_fc, blur_kernel=blur_kernel
        )
        self.to_rgb1 = CoLoRToRGB(self.channels[4], style_dim,
                                     rank=rank, use_act=use_act,
                                     lifelong=lifelong, lora_alpha_conv = lora_alpha_conv, lora_alpha_fc = lora_alpha_fc,
                                     upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                CoLoRStyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    rank=rank,
                    use_act=use_act,
                    lifelong=lifelong,
                    lora_alpha_conv = lora_alpha_conv,
                    lora_alpha_fc = lora_alpha_fc,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                CoLoRStyledConv(
                    out_channel,
                    out_channel,
                    3,
                    style_dim,
                    rank=rank,
                    use_act=use_act,
                    lifelong=lifelong,
                    lora_alpha_conv = lora_alpha_conv,
                    lora_alpha_fc=lora_alpha_fc,
                    blur_kernel=blur_kernel,
                )
            )

            self.to_rgbs.append(
                CoLoRToRGB(out_channel, style_dim,
                              rank=rank, use_act=use_act, lora_alpha_conv=lora_alpha_conv, lora_alpha_fc=lora_alpha_fc,
                              lifelong=lifelong)
            )

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        return_feats=False,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        feat_list = []
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        feat_list.append(out)

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            feat_list.append(out)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            feat_list.append(out)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent
        elif return_feats:
            return image, feat_list
        else:
            return image, None


class CoLoRConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        rank,
        use_act,
        lifelong,
        lora_alpha_conv,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            CoLoREqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                rank=rank,
                use_act=use_act,
                lora_alpha_conv = lora_alpha_conv,
                lifelong=lifelong,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class CoLoRResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, rank, use_act, lifelong, lora_alpha_conv, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = CoLoRConvLayer(in_channel, in_channel, 3,
                                       rank=rank, use_act=use_act, lifelong=lifelong, lora_alpha_conv=lora_alpha_conv)
        self.conv2 = CoLoRConvLayer(in_channel, out_channel, 3,
                                       rank=rank, use_act=use_act, lifelong=lifelong, lora_alpha_conv=lora_alpha_conv,downsample=True)

        self.skip = CoLoRConvLayer(
            in_channel, out_channel, 1, rank=rank, use_act=use_act, lifelong=lifelong, lora_alpha_conv=lora_alpha_conv,
            downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class CoLoRDiscriminator(nn.Module):
    def __init__(self, size, rank, use_act, lifelong, lora_alpha_conv, lora_alpha_fc,
                 channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.lfs = lifelong
        if lifelong is True:
            print("*"*20)
            print("Use CoLoR learning for Discriminator")

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [CoLoRConvLayer(3, channels[size], 1,
                                   rank=rank, use_act=use_act, lifelong=lifelong, lora_alpha_conv = lora_alpha_conv)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(CoLoRResBlock(in_channel, out_channel,
                                          rank=rank, use_act=use_act, lifelong=lifelong, lora_alpha_conv=lora_alpha_conv,
                                          blur_kernel=blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = CoLoRConvLayer(in_channel + 1, channels[4], 3,
                                            rank=rank, use_act=use_act, lifelong=False, lora_alpha_conv=lora_alpha_conv,)
        self.final_linear = nn.Sequential(
            CoLoREqualLinear(channels[4] * 4 * 4, channels[4],
                             rank=rank, lifelong=False,
                             activation="fused_lrelu"),
            CoLoREqualLinear(channels[4], 1,
                             rank=rank, lifelong=False,),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out, None


class CoLoRPatchDiscriminator(nn.Module):
    def __init__(self, size, rank=None, use_act=False, lifelong=False, lora_alpha_conv=1.0, 
                 lora_alpha_fc = 1.0, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.lfs = lifelong

        convs = [CoLoRConvLayer(3, channels[size], 1,
                                   rank=rank, use_act=use_act, lifelong=lifelong, lora_alpha_conv=lora_alpha_conv)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(CoLoRResBlock(in_channel, out_channel,
                                          rank=rank, use_act=use_act, lifelong=lifelong,
                                          blur_kernel=blur_kernel, lora_alpha_conv = lora_alpha_conv))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)
        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = CoLoRConvLayer(in_channel + 1, channels[4], 3,
                                         rank=rank, use_act=use_act, lifelong=lifelong, 
                                         lora_alpha_conv=lora_alpha_conv)
        self.final_linear = nn.Sequential(
            CoLoREqualLinear(channels[4] * 4 * 4, channels[4], 
                             rank=rank, lifelong=False, lora_alpha_fc=lora_alpha_fc, activation='fused_lrelu'),
            CoLoREqualLinear(channels[4], 1, 
                             rank=rank, lora_alpha_fc=lora_alpha_fc, lifelong=False),
        )
        
    def forward(self, inp, extra=None, flag=None, p_ind=None):

        feat = []
        for i in range(len(self.convs)):
            if i == 0:
                inp = self.convs[i](inp)
            else:
                temp1 = self.convs[i].conv1(inp)
                if (flag > 0) and (temp1.shape[1] == 512) and (temp1.shape[2] == 32 or temp1.shape[2] == 16):
                    feat.append(temp1)
                temp2 = self.convs[i].conv2(temp1)
                if (flag > 0) and (temp2.shape[1] == 512) and (temp2.shape[2] == 32 or temp2.shape[2] == 16):
                    feat.append(temp2)
                inp = self.convs[i](inp)
                if (flag > 0) and len(feat) == 4:
                    # We use 4 possible intermediate feature maps
                    # to be used for patch-based adversarial loss.
                    # Any one of them is selected randomly during training.
                    inp = extra(feat[p_ind], p_ind)
                    return inp, None

        out = inp
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        feat.append(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)
        return out, None


class Extra(nn.Module):
    def __init__(self, rank=None, use_act=False, lifelong=False, lora_alpha_conv = 1.0):
        super().__init__()
        self.lfs = lifelong
        self.new_conv = nn.ModuleList()
        self.new_conv.append(CoLoRConvLayer(512, 1, 3, rank=rank, use_act=use_act, lifelong=False, lora_alpha_conv=lora_alpha_conv))
        self.new_conv.append(CoLoRConvLayer(512, 1, 3, rank=rank, use_act=use_act, lifelong=False, lora_alpha_conv=lora_alpha_conv))
        self.new_conv.append(CoLoRConvLayer(512, 1, 3, rank=rank, use_act=use_act, lifelong=False, lora_alpha_conv=lora_alpha_conv))
        self.new_conv.append(CoLoRConvLayer(512, 1, 3, rank=rank, use_act=use_act, lifelong=False, lora_alpha_conv=lora_alpha_conv))

    def forward(self, inp, ind):
        out = self.new_conv[ind](inp)
        return out