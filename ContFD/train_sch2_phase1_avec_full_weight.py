from GPUtil import showUtilization as gpu_usage

gpu_usage()

import time
import argparse
import os
import torch
import csv
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model_classify import Generator as Model3D
from  model_weight import Weight as WeightDepth
from  model_full_weight import Weight as WeightDepth_cat
from loss import ContrastiveLoss, ssim
from data_5 import CsvDataset
from torch.optim import lr_scheduler
from utils import AverageMeter


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=16, type=int, help='batch size')
    args = parser.parse_args()
    
    country_name = 'iva'

    train_csv_dir = './data/training_20_samples/training_' + country_name + '.csv'

    train_me_where = "from_beginning"  # "from_middle"

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    # Create model
    weightDepth = WeightDepth().to(device)
    weightDepth_cat = WeightDepth_cat().to(device)
    model3D = Model3D().to(device)
    print('Model created.')
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        # model = nn.DataParallel(model.cuda())
        model3D = nn.DataParallel(model3D.to(device))

    print('model and cuda mixing done')

    # Training parameters
    optimizer = torch.optim.Adam(model3D.parameters(), args.lr)
    my_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_loss = 100.0
    batch_size = args.bs

    # Load data
    transform1 = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])
    transform2 = transforms.Compose([transforms.Resize((120, 120)), transforms.ToTensor()])
    training_dataset = CsvDataset(csv_file=train_csv_dir, transform1=transform1, transform2=transform2,
                                  should_invert=False)
    train_loader = DataLoader(training_dataset, shuffle=True, num_workers=4 * torch.cuda.device_count(),
                              batch_size=batch_size, pin_memory=True)

    print('train_loader', len(train_loader))

    # Loss
    criterion_contrastive = ContrastiveLoss()
    l1_criterion = nn.L1Loss()

    print("Total number of batches in train loader are :", len(train_loader))

    if train_me_where == "from_middle":
        checkpoint = torch.load('./checkpoint/' + '/Models' + '.ckpt')
        model3D.load_state_dict(checkpoint['state_dict'])

    # writer = SummaryWriter('./runs/real_siamese_net_running')


    # Start training...
    for epoch in range(args.epochs):

        losses = AverageMeter()
        N = len(train_loader)

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Switch to train mode
        model3D.train()

        total_batch_loss = 0.0

        for i, data in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()

            # Prepare sample and target
            img1, img1_half, img2, img2_half, label_1, label_2 = data  # label_0 is the label of image_0 and label_1 is for image_1
            img1, img1_half, img2, img2_half, label_1, label_2 = img1.to(device), img1_half.to(device), img2.to(
                device), img2_half.to(device), label_1.to(device), label_2.to(device)

            label_pair = label_1[:] == label_2[:]  # label of pairs: 1 if the two images in the pair are
            # of the same class, 0 if the images belong to two different classes
            label_pair = label_pair.long()
            latent_1, decoder_op_1 = model3D(img1)
            latent_2, decoder_op_2 = model3D(img2)
                        
            weight_feature_depth_1 = weightDepth(latent_1)  
            weight_feature_depth_2 = weightDepth(latent_2)
            latent_cat = torch.cat((latent_1,latent_2), 1) 
            weight_feature_depth_cat = weightDepth_cat(latent_cat) 

            # loss_contrastive_siamese_latent = criterion_contrastive(output1, output2, label_pair)
            loss_contrastive_densNet_latent = criterion_contrastive(latent_1, latent_2, label_pair)

            # Compute the loss for input image into the encoder and the output from the decoder
            loss_image_direct_1 = l1_criterion(decoder_op_1, img1_half)
            loss_ssim_image_direct_1 = torch.clamp((1 - ssim(decoder_op_1.float(), img1_half.float(), val_range=1)) * 0.5, 0, 1)
            loss_image_total_1 = weight_feature_depth_1[0]* loss_image_direct_1 + (1-weight_feature_depth_1[0])* loss_ssim_image_direct_1

            loss_image_direct_2 = l1_criterion(decoder_op_2, img2_half)
            loss_ssim_image_direct_2 = torch.clamp((1 - ssim(decoder_op_2.float(), img2_half.float(), val_range=1)) * 0.5, 0, 1)  # clamps all the input elements into the range [ min, max ]
            loss_image_total_2 = weight_feature_depth_2[0]* loss_image_direct_2 + (1-weight_feature_depth_2[0])* loss_ssim_image_direct_2

            # loss_total = loss_contrastive_siamese_latent + loss_contrastive_densNet_latent + loss_image_total_1 + loss_image_total_2
            loss_total = weight_feature_depth_cat[0] * loss_contrastive_densNet_latent + weight_feature_depth_cat[1] * loss_image_total_1 + weight_feature_depth_cat[2] * loss_image_total_2

            print('loss_total-----------------------------', loss_total)

            loss_total.backward()
            optimizer.step()

            # Update step
            total_batch_loss = loss_total
            losses.update(total_batch_loss.data.item(), img1.size(0))

            torch.cuda.empty_cache()
        # Log progress; print after every epochs into the console
        print('Epoch: [{:.4f}] \t The loss of this epoch is: {:.4f} '.format(epoch, losses.avg))

        if losses.avg < best_loss:
            print("Here the training loss got reduced, hence printing")
            print('Current best epoch loss is {:.4f}'.format(losses.avg), 'previous best was {}'.format(best_loss))
            best_loss = losses.avg
            _save_best_model(model3D, best_loss, epoch, country_name)

        # save the losses avg in .csv file
        with open("loss_avg_sch2_phase1_avec_full_weight_" + country_name + ".csv", 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, losses.avg])

        my_lr_scheduler.step()


def _save_best_model(model, best_loss, epoch, country_name):
    # Save Model

    state = {
        'state_dict': model.state_dict(),
        'best_acc': best_loss,
        'cur_epoch': epoch
    }

    if not os.path.isdir('./checkpoint/'):
        os.makedirs('./checkpoint/' )

    torch.save(state,
               './checkpoint/' + 'sch2_phase1_avec_full_weight_' + country_name + '.ckpt')


if __name__ == '__main__':
    main()
