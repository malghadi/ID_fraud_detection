import torch
from math import exp
import torch.nn.functional as F
import math
import sys
import numpy as np
import torch.nn as nn


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):

    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        # ret = ssim_map[~torch.isnan(ssim_map)].mean()
        ret = ssim_map.nanmean()

    else:
        # ret = ssim_map[~torch.isnan(ssim_map)].mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if math.isnan(ret) or math.isinf(ret):
        print("Check me, I have issues")
        print("The value of ret is {}".format(ret))

        if ssim_map.isnan().any():
            print("Print ssim_map : ", ssim_map)
        if img1.isnan().any():
            print("Print pred image : ", img1)
        if img2.isnan().any():
            print("Print original image : ", img2)

        sys.exit("Due to NaN value, I am exiting")

    if full:
        return ret, cs

    return ret


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_1, output_2, label_lb):
        euclidean_distance = F.pairwise_distance(output_1, output_2, keepdim=True)
        # contrastive_loss = torch.mean((1 - label_lb) * torch.pow(euclidean_distance, 2) +
        #                               label_lb * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        contrastive_loss = torch.mean(label_lb * torch.pow(euclidean_distance,2) + (1 - label_lb) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return contrastive_loss


class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='hinge', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge': ## hinge loss (loss for training Discriminator and Generator are different)
            if is_disc:
                ## for training Discriminator: loss = E(Relu(1-D(real)))+E(Relu(1+D(fake))), here outputs = D(inputs)
                ## i.e.,  mean(D(real))->1, mean(D(fake))-> -1, mean is E()
                ## If the D is already good enough that mean(D(real)) > 1 or mean(D(fake))<-1, we don't need the training anymore.
                ## Thus, we only train D when its output 1>mean(D(input))>-1.
                if is_real: ## inputing real images
                    outputs = -outputs
                ## mean value of ReLU(1+output).
                ## ReLU is an element-wise operator which won't change the shape of output
                return self.criterion(1 + outputs).mean()
            else:
                ## for training Generator: loss = - E(D(fake)) = E(-D(fake)). For a trained Discriminator D, D(fake) should be -1, D(real) should be 1.
                ## Thus, here training the Generator to output a fake image being good enought to confuses the Discriminator, making D(fake) = 0 instead of -1.
                ## N.B, hinge loss doesn't define the loss = E(1-D(fake)) which would push the Generator to generate an extremely good image which makes the D(input)->1,
                ## i.e. the Discriminator would definately consider the input is a real image instead of a fake one.
                ## Since when D(input)=0, it means that the input is good enough to confuse the Discriminator and it's time to retrain the Discriminator.
                ## That is to say, we don't have to wait the Discriminator being totally lost and then retrain the Discriminator.
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(
                outputs)
            loss = self.criterion(outputs, labels)
            return loss