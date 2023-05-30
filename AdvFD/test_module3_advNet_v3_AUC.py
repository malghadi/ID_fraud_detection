import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Generator, Discriminator 
import argparse
from data_5 import CsvDataset
import csv
import statistics
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    args = parser.parse_args()
    batch_size = args.bs
 
    country_list = ['alb','aze','esp','est','fin','grc','iva','rus','srb','svk']
    for country_name in country_list:

        auc_list_country = []
        fpr_list = []
        tpr_list = []

        test_csv_dir = './data/testing/testing_advNet_v3_' + country_name + '.csv'

        # Load data
        transform1 = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])
        transform2 = transforms.Compose([transforms.Resize((120, 120)), transforms.ToTensor()])
        testing_dataset = CsvDataset(csv_file=test_csv_dir, transform1=transform1, transform2=transform2, should_invert=False)
        test_loader = DataLoader(testing_dataset, shuffle=True, num_workers=4 * torch.cuda.device_count(), batch_size=batch_size, pin_memory=True)
        print('test_loader', len(test_loader))

        is_use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if is_use_cuda else "cpu")

        #################################
        netG = Generator().to(device)
        netD = Discriminator().to(device)
        
        # path = './checkpoint/siamese_net/module_3_advNet_v3_' + country_name + '.ckpt' 
        path = './checkpoint/siamese_net/advNet_full_weight_' + country_name + '.ckpt' 

        print(path)
        checkpoint = torch.load(path, map_location=device)
        dis_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint['netD_state_dict'].items()}
        netD.load_state_dict(dis_loaded_state)
        gen_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint['netG_state_dict'].items()}
        netG.load_state_dict(gen_loaded_state)
        netD.eval()
        netG.eval()
        ################
        test_lbl = []
        test_score = []
        auc_list = [] 

        testing_data_length = [15,30,45,60,75,90,105,120] 
        for testing_length in testing_data_length:
            for epoch in range(1):           
                for i, data in enumerate(test_loader):
                    if i == testing_length:
                        break
                    # Prepare sample and target
                    img1, img1_half, img2, img2_half, label_image_1, label_image_2 = data  # label_0 is the label of image_0 and label_1 is for image_1
                    img1, img1_half, img2, img2_half, label_image_1, label_image_2 = img1.to(device), img1_half.to(device), img2.to(
                            device), img2_half.to(device), label_image_1.to(device), label_image_2.to(device)

                    latent_1, decoder_op_1 = netG(img1) # img1 is always true (real) image, img2 unknown
                                                                                    # we are testing to know class of img2 
                    latent_2, decoder_op_2 = netG(img2)
                    real_vid_feat, y1o_softmax  = netD(latent_1)
                    fake_vid_feat, y2o_softmax = netD(latent_2)  
                    
                    test_lbl.append(int(label_image_1.item()))
                    test_score.append(torch.max(y1o_softmax).item())
                    
                    test_lbl.append(int(label_image_2.item()))
                    test_score.append(torch.max(y2o_softmax).item())               
                
            fpr, tpr, _ = roc_curve(test_lbl, test_score)
            
            auc_list.append(metrics.auc(fpr, tpr)) 
        
        # fpr_list.append(fpr)
        # tpr_list.append(tpr)

        if not os.path.isdir('./results'):
                os.makedirs('./results') 
    
        with open("results/auc.csv", 'a') as file:
            writer = csv.writer(file)
            for myList in auc_list:
                file.write(str(myList))
                file.write (',')
                #file.write(str(auc_list) )
            file.write ('\n')
        
        # for indx in range(len(fpr_list)):
        #     plt.plot(fpr_list[indx], tpr_list[indx], label = testing_length[indx])
        #     plt.xlabel('False Positive Rate')
        #     plt.ylabel('True Positive Rate')
        #     plt.xlim([0.0, 1.0])
        #     plt.ylim([0.0, 1.05])
        #     plt.legend()
        #     plt.title('Receiver operating characteristic')
        #     plt.savefig('ROC_' + country_name + '.png')

if __name__ == '__main__':
    main()

    