from GPUtil import showUtilization as gpu_usage
gpu_usage()
import argparse
import os
import torch
import csv
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model import Generator, Discriminator 
from loss import ContrastiveLoss
from data_5 import CsvDataset
from torch.optim import lr_scheduler
from utils import AverageMeter


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    args = parser.parse_args()
    
    country_name = 'svk'

    train_csv_dir = './data/training_20_samples/training_v1_' + country_name + '.csv'

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    # Create model
    model = Discriminator().to(device)
    print('Model created.')
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model.to(device))

    model3D = Generator().to(device)
    model3D.load_state_dict(torch.load('./checkpoint/siamese_net/module_1_sch2_phase1_' + country_name + '.ckpt'), strict=False)
    for param in model3D.parameters():
        param.requires_grad = True
    
    #Model3D.load_state_dict(torch.load('./checkpoint/siamese_net/model3D.ckpt', map_location=device))
    print('model and cuda mixing done')

    # Training parameters
    optimizer_model = torch.optim.Adam(model.parameters(), args.lr)
    my_lr_scheduler_model = lr_scheduler.StepLR(optimizer_model, step_size=20, gamma=0.1)
    optimizer_model3D = torch.optim.Adam(model3D.parameters(), args.lr)
    my_lr_scheduler_model3D = lr_scheduler.StepLR(optimizer_model3D, step_size=20, gamma=0.1)
    

    best_acc = 0.0
    batch_size = args.bs

    # Load data
    transform1 = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])
    transform2 = transforms.Compose([transforms.Resize((120, 120)), transforms.ToTensor()])
    training_dataset = CsvDataset(csv_file=train_csv_dir, transform1=transform1, transform2=transform2,
                                  should_invert=False)

    train_loader = DataLoader(training_dataset, shuffle=True, num_workers=4 * torch.cuda.device_count(),
                              batch_size=batch_size, pin_memory=True)

    dataset_sizes = len(train_loader.dataset)

    # Loss
    criterion_contrastive = ContrastiveLoss()
    l1_criterion = nn.L1Loss()
    m = nn.Sigmoid()
    loss_cross = torch.nn.CrossEntropyLoss()

    print("Total number of batches in train loader are :", len(train_loader))

    # writer = SummaryWriter('./runs/real_siamese_net_running')

    # Start training...
    for epoch in range(args.epochs):

        losses = AverageMeter()
        N = len(train_loader)

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Switch to train mode
        model.train()

        total_batch_loss = 0.0
        get_corrects = 0.0

        for i, data in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                optimizer_model.zero_grad()
                optimizer_model3D.zero_grad()

            # Prepare sample and target
            img1, img1_half, img2, img2_half, label_image_1, label_image_2 = data  # label_0 is the label of image_0 and label_1 is for image_1
            img1, img1_half, img2, img2_half, label_image_1, label_image_2 = img1.to(device), img1_half.to(device), img2.to(
                device), img2_half.to(device), label_image_1.to(device), label_image_2.to(device)

            label_pair = label_image_1[:] == label_image_2[:]  # label of pairs: 1 if the two images in the pair are
            # of the same class, 0 if the images belong to two different classes
            label_pair = label_pair.long()
            latent_1, latent_2, decoder_op_1, decoder_op_2 = model3D(img1,img2)
            
            class_op_1, y1o_softmax = model(latent_1)
            class_op_2, y2o_softmax = model(latent_2)

            loss_cross_entropy_1 = loss_cross(class_op_1, label_image_1)
            loss_cross_entropy_2 = loss_cross(class_op_2, label_image_2) 
            
            # get_corrects += torch.sum(torch.max(y1o_softmax, 1)[1] == label_image_1) # in this case, you need to use the soft_max values to get the maximum probs
            # get_corrects += torch.sum(torch.max(y2o_softmax, 1)[1] == label_image_2) # in this case, you need to use the soft_max values to get the maximum probs
         
            get_corrects += torch.sum(torch.logical_and(torch.max(y1o_softmax, 1)[1] == label_image_1, torch.max(y2o_softmax, 1)[1] == label_image_2))

            # loss_total = loss_contrastive_siamese_latent + loss_contrastive_densNet_latent + loss_image_total_1 + loss_image_total_2
            loss_total =  loss_cross_entropy_1 + loss_cross_entropy_2
            print('loss_total-----------------------------', loss_total)

            loss_total.backward()
            optimizer_model.step()
            optimizer_model3D.step()


            # Update step
            total_batch_loss = loss_total
            losses.update(total_batch_loss.data.item(), img1.size(0))

            torch.cuda.empty_cache()
        
        variable_acc = get_corrects.item() / dataset_sizes 

        # Log progress; print after every epochs into the console
        print('Epoch: [{:.4f}] \t The loss of this epoch is: {:.4f} \t The accuracy of this epoch is: {:.4f} '.format(epoch, losses.avg, variable_acc))

        # Log to tensorboard
        # writer_1.add_scalar('Train/Each Epoch Loss', epoch_loss, niter)

        if variable_acc > best_acc:
            print("Here the training accuracy got reduced, hence printing")
            print('Current best epoch accuracy is {:.4f}'.format(variable_acc), 'previous best was {}'.format(best_acc))
            best_acc = variable_acc
            model_name = "siamese_net"                      
            torch.save({
                'model_state_dict': model.state_dict(),
                'model3D_state_dict': model3D.state_dict()},
                './checkpoint/' + model_name + '/module_1_sch3_' + country_name  + '.ckpt')

        # save the losses avg in .csv file
        with open("loss_avg_module_1_sch3_" + country_name + ".csv", 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, losses.avg, variable_acc])
        
        
        my_lr_scheduler_model.step()
        my_lr_scheduler_model3D.step()


if __name__ == '__main__':
    main()
