import gc
from operator import ne
from GPUtil import showUtilization as gpu_usage
import matplotlib.pyplot as plt
import statistics
import time
import argparse
import datetime
import os
import numpy as np
import torch
import sys
import csv
import torch.nn as nn
import torchvision.utils as vutils
from torchvision import transforms
import torchvision.datasets as dataset
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from model import Generator, Discriminator 
from loss import ContrastiveLoss, ssim, AdversarialLoss
from data_5 import SiameseNetworkDataset, CsvDataset
from torch.optim import lr_scheduler
from utils import AverageMeter

gpu_usage()

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--beta1', default=0.5, type=float, help='hyperparam for Adam optimizers')

    args = parser.parse_args()
    
    country_name = 'srb'

    train_csv_dir = './data/training_20_samples/training_advNet_v3_' + country_name + '.csv'
    # validation_csv_dir = './data/validation/validation_alb.csv'

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")
    ngpu = torch.cuda.device_count()

    # Create the Discriminator
    netD = Discriminator().to(device)
    print('Discriminator Model created.')
    if (device.type == 'cuda') and (ngpu > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netD = nn.DataParallel(netD.to(device))
    # print(netD)


    # Create the Generator
    netG = Generator().to(device)
    print('Generator Model created.')
    if (device.type == 'cuda') and (ngpu > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG = nn.DataParallel(netG.to(device))
    # print(netG)
   

    print('model and cuda mixing done')

    # Training parameters
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    D_lr_scheduler = lr_scheduler.StepLR(optimizerD, step_size=25, gamma=0.1)
    G_lr_scheduler = lr_scheduler.StepLR(optimizerG, step_size=25, gamma=0.1)

    best_loss = 100.0
    best_acc = 0.0
    batch_size = args.bs

    # Load data
    transform1 = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])
    transform2 = transforms.Compose([transforms.Resize((120, 120)), transforms.ToTensor()])
    training_dataset = CsvDataset(csv_file=train_csv_dir, transform1=transform1, transform2=transform2,
                                  should_invert=False)
    # validation_dataset = CsvDataset(csv_file=validation_csv_dir, transform=transform, should_invert=False)

    train_loader = DataLoader(training_dataset, shuffle=True, num_workers=1 * torch.cuda.device_count(),
                              batch_size=batch_size, pin_memory=True)
    # validation_loader = DataLoader(validation_dataset, shuffle=True, num_workers=6, batch_size=batch_size)
 
    print("Total number of batches in train loader are :", len(train_loader))
    dataset_sizes = len(train_loader.dataset)
    print('dataset_sizes', dataset_sizes)


    # Loss
    criterion_contrastive = ContrastiveLoss()
    l1_criterion = nn.L1Loss()
    loss_cross = torch.nn.CrossEntropyLoss()
    bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
    adversarial_loss = AdversarialLoss()
    adversarial_loss = adversarial_loss.to(device)

    writer = SummaryWriter('./runs/real_siamese_net_running')

    # real_imag_label = 1
    # fake_imag_label = 0
    # d_label_real_img = torch.full((batch_size, 1), real_imag_label, device=device)
    # d_label_fake_img = torch.full((batch_size, 1), fake_imag_label, device=device)
    # g_label_cheat_real_img = torch.full((batch_size, 1), real_imag_label, device=device)
    
    real_imag_label = 1
    fake_imag_label = 0
    d_label_real_img = torch.cuda.LongTensor([1]*batch_size)
    d_label_fake_img = torch.cuda.LongTensor([0]*batch_size)
    g_label_fake_img = torch.cuda.LongTensor([1]*batch_size)


    # Start training...
    for epoch in range(args.epochs):
        # torch.cuda.empty_cache()

        losses = AverageMeter()
        N = len(train_loader)

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Switch to train mode
        # model.train()

        keep_all_batch_losses = []
        total_batch_loss = 0.0
        running_batch_loss = 0.0

        G_losses = []
        D_losses = []
        get_corrects = 0.0

        for i, data in enumerate(train_loader):

            # netD.train()
            # netG.train() 
            gen_loss = 0
            dis_loss = 0
            # with torch.enable_grad():   

            # Prepare sample and target
            img1, img1_half, img2, img2_half, label_image_1, label_image_2 = data  # label_0 is the label of image_0 and label_1 is for image_1
            img1, img1_half, img2, img2_half, label_image_1, label_image_2 = img1.to(device), img1_half.to(device), img2.to(
                device), img2_half.to(device), label_image_1.to(device), label_image_2.to(device)

            label_pair = label_image_1[:] == label_image_2[:]  # label of pairs: 1 if the two images in the pair are
            # of the same class, 0 if the images belong to two different classes
            label_pair = label_pair.long()
            
            latent_1, latent_2, decoder_op_1, decoder_op_2 = netG(img1, img2)

            ############################
            # Calculate the contrastive loss
            ############################
            # contrastive_loss_densNet = criterion_contrastive(latent_1, latent_2, label_pair)
            
            ############################
            # Calculate the ssim loss   # (the loss for input image into the encoder and the output from the decoder)
            loss_image_direct_1 = l1_criterion(decoder_op_1, img1_half)
            # print('loss_image_direct_1-----------------------------', loss_image_direct_1)
            loss_ssim_image_direct_1 = torch.clamp((1 - ssim(decoder_op_1.float(), img1_half.float(), val_range=1)) * 0.5, 0, 1) # torch. clamp() is used to clamp (restrict) all the elements in an input into the range [min, max]. 
                                                                                                                                # It takes three parameters: the input tensor, min, and max values.
            loss_image_total_1 = (1.0 * loss_image_direct_1) + (0.1 * loss_ssim_image_direct_1)
            # print('loss_image_total_1-----------------------------', loss_image_total_1)

            loss_image_direct_2 = l1_criterion(decoder_op_2, img2_half)
            loss_ssim_image_direct_2 = torch.clamp((1 - ssim(decoder_op_2.float(), img2_half.float(), val_range=1)) * 0.5, 0, 1)  
            loss_image_total_2 = (1.0 * loss_image_direct_2) + (0.1 * loss_ssim_image_direct_2)
            # print('loss_image_total_2-----------------------------', loss_image_total_2)

            ############################
            # (1) Update D network: maximize log(D(x)))     # discriminator adversarial loss
            ###########################
            # Calculate loss on all-real batch
            # netD.zero_grad()
            optimizerD.zero_grad()
            real_vid_feat, y1o_softmax  = netD(latent_1)
            # h = torch.max(y1o_softmax, 1)[1]
            dis_real_loss = loss_cross(real_vid_feat, d_label_real_img) # 1
            # dis_real_loss = adversarial_loss(real_vid_feat, True, True)
            dis_real_loss.backward(retain_graph=True)


            # Calculate loss on all-fake batch
            fake_vid_feat, y2o_softmax = netD(latent_2.detach()) 
            # h2 = torch.max(y2o_softmax, 1)[1]       
            dis_fake_loss = loss_cross(fake_vid_feat, d_label_fake_img)
            # dis_fake_loss = adversarial_loss(fake_vid_feat, False, True)
            dis_fake_loss.backward(retain_graph=True) 
            # dis_loss = (dis_real_loss + dis_fake_loss)/2  # the gradients accumulated from both the all-real and all-fake batches
            #dis_loss += contrastive_loss_densNet + loss_image_total_1 + loss_image_total_2
            #dis_loss += contrastive_loss_densNet

            D_losses.append(dis_fake_loss.item())

            #dis_loss.backward(retain_graph=True)
            # Update D
            optimizerD.step()
                        
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            # generator adversarial loss
            ###########################
            
            # netG.zero_grad()
            optimizerG.zero_grad()
           
            gen_fake_feat, y2o_softmax_fake = netD(latent_2)
            # gen_fake_loss = loss_cross(gen_fake_feat, d_label_real_img)
            gen_fake_loss = loss_cross(gen_fake_feat, g_label_fake_img)
            # gen_fake_loss = loss_cross(gen_fake_feat, d_label_fake_img)
            # gen_fake_loss = adversarial_loss(gen_vid_feat, True, False)
            
                        
            # gen_loss = (gen_fake_loss + contrastive_loss_densNet + loss_image_total_1 + loss_image_total_2)/4
            # gen_loss =  gen_fake_loss + loss_image_total_1 
            gen_loss =  (gen_fake_loss + loss_image_total_1 + loss_image_total_2)/3
            # gen_loss =  (gen_fake_loss + loss_image_total_2)/2
            G_losses.append(gen_loss.item())
            # gen_loss_total = gen_loss +  loss_image_total_1 + loss_image_total_2
            # loss_total = contrastive_loss_densNet + loss_image_total_1 + loss_image_total_2            
            gen_loss.backward()
            # Update G
            optimizerG.step()
            
            get_corrects += torch.sum(torch.logical_and(torch.max(y1o_softmax, 1)[1] == label_image_1, torch.max(y2o_softmax, 1)[1] == label_image_2))
            print('get_corrects', get_corrects)

            # print('loss_total---',  dis_loss, gen_loss)

        variable_acc = get_corrects.item() / dataset_sizes

        # print('Epoch: [{:.4f}] \t The loss of this epoch is: {:.4f} \t The accuracy of this epoch is: {:.4f} '.format(epoch, losses.avg, variable_acc))
        print('Epoch: [{:.4f}] \t The dis_loss of this epoch is: {:.4f} \t The gen_loss of this epoch is: {:.4f}\t The accuracy of this epoch is: {:.4f} '.format(epoch, statistics.mean(D_losses), statistics.mean(G_losses), variable_acc))

        if variable_acc > best_acc:
            print("Here the training accuracy got reduced, hence printing")
            print('Current best epoch accuracy is {:.4f}'.format(variable_acc), 'previous best was {}'.format(best_acc))
            best_acc = variable_acc
            model_name = "siamese_net"                      
            torch.save({
                'netD_state_dict': netD.state_dict(),
                'netG_state_dict': netG.state_dict()},
                './checkpoint/' + model_name + '/module_3_advNet_v3_' + country_name  + '.ckpt')
            

        # save the losses avg in .csv file
        with open("loss_avg_module_3_advNet_v3_" + country_name + ".csv", 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, statistics.mean(D_losses), statistics.mean(G_losses), variable_acc])
        
       
        D_lr_scheduler.step()
        G_lr_scheduler.step()



if __name__ == '__main__':
    main()