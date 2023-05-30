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
from model_classify import Generator, Discriminator 
from loss import ContrastiveLoss
from data_5 import CsvDataset
from torch.optim import lr_scheduler
from utils import AverageMeter
from  model_weight import Weight as WeightDepth



def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    args = parser.parse_args()
    
    country_name = 'alb'

    train_csv_dir = './data/training_20_samples/training_' + country_name + '.csv'

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    # Create model
    weightDepth = WeightDepth().to(device)
    model = Discriminator().to(device)
    print('Model created.')
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model.to(device))

    model3D = Generator().to(device)
    model3D.load_state_dict(torch.load('./checkpoint/sch2_phase1_avec_weight_' + country_name + '.ckpt'), strict=False)
    for param in model3D.parameters():
        param.requires_grad = False
    
    #Model3D.load_state_dict(torch.load('./checkpoint/siamese_net/model3D.ckpt', map_location=device))
    print('model and cuda mixing done')

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    my_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

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
                optimizer.zero_grad()

            # Prepare sample and target
            img1, img1_half, img2, img2_half, label_image_1, label_image_2 = data  # label_0 is the label of image_0 and label_1 is for image_1
            img1, img1_half, img2, img2_half, label_image_1, label_image_2 = img1.to(device), img1_half.to(device), img2.to(
                device), img2_half.to(device), label_image_1.to(device), label_image_2.to(device)

            label_pair = label_image_1[:] == label_image_2[:]  # label of pairs: 1 if the two images in the pair are
            # of the same class, 0 if the images belong to two different classes
            label_pair = label_pair.long()
            latent_1, decoder_op_1 = model3D(img1)
            latent_2, decoder_op_2 = model3D(img2)
            
            class_op_1, y1o_softmax = model(latent_1)
            class_op_2, y2o_softmax = model(latent_2)

            weight_feature_depth_1 = weightDepth(latent_1)  
            # weight_feature_depth_2 = weightDepth(latent_2)

            loss_cross_entropy_1 = loss_cross(class_op_1, label_image_1)
            loss_cross_entropy_2 = loss_cross(class_op_2, label_image_2) 
            
            # get_corrects += torch.sum(torch.max(y1o_softmax, 1)[1] == label_image_1) # in this case, you need to use the soft_max values to get the maximum probs
            # get_corrects += torch.sum(torch.max(y2o_softmax, 1)[1] == label_image_2) # in this case, you need to use the soft_max values to get the maximum probs
         
            get_corrects += torch.sum(torch.logical_and(torch.max(y1o_softmax, 1)[1] == label_image_1, torch.max(y2o_softmax, 1)[1] == label_image_2))

            # loss_total = loss_contrastive_siamese_latent + loss_contrastive_densNet_latent + loss_image_total_1 + loss_image_total_2
            loss_total =  weight_feature_depth_1[0] * loss_cross_entropy_1 + (1-weight_feature_depth_1[0]) * loss_cross_entropy_2
            print('loss_total-----------------------------', loss_total)

            loss_total.backward()
            optimizer.step()


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
            _save_best_model(model, best_acc, epoch, country_name)

        # save the losses avg in .csv file
        with open("loss_avg_sch2_phase2_avec_weight_v2_" + country_name + ".csv", 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, losses.avg, variable_acc])
        my_lr_scheduler.step()


def _save_best_model(model, best_acc, epoch, country_name):
    # Save Model
    state = {
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'cur_epoch': epoch
    }
    if not os.path.isdir('./checkpoint/' ):
        os.makedirs('./checkpoint/' )

    torch.save(state,
               './checkpoint/' +
                'sch2_phase2_avec_weight_v2_' + country_name + '.ckpt')

if __name__ == '__main__':
    main()
