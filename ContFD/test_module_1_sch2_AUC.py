import torch
import statistics
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model_classify import Discriminator 
from model_classify import Generator as Model3D
import argparse
from data_5 import CsvDataset
import csv
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import os 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    args = parser.parse_args()
    batch_size = args.bs
    
    country_list = ['alb', 'aze', 'esp', 'est', 'fin', 'grc', 'iva', 'rus', 'srb', 'svk']    
    for country_name in country_list:    
        F_score_list = []
        
        test_csv_dir = './data/testing/testing_' + country_name + '.csv' 

        # Load data
        transform1 = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])
        transform2 = transforms.Compose([transforms.Resize((120, 120)), transforms.ToTensor()])
        testing_dataset = CsvDataset(csv_file=test_csv_dir, transform1=transform1, transform2=transform2, should_invert=False)
        test_loader = DataLoader(testing_dataset, shuffle=True, num_workers=4 * torch.cuda.device_count(), batch_size=batch_size, pin_memory=True)
        print('test_loader', len(test_loader))

        is_use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if is_use_cuda else "cpu")

        #################
        model = Discriminator().to(device)

        model3D = Model3D().to(device)
        
        path_model = './checkpoint/sch2_phase2_avec_full_weight_' + country_name + '.ckpt'
        checkpoint1 = torch.load(path_model, map_location=device)
        dis_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint1['state_dict'].items()}
        model.load_state_dict(dis_loaded_state)
        model.eval()
        
        # path_model3D = './checkpoint/sch2_phase1_avec_weight_' + country_name + '.ckpt'
        # checkpoint2 = torch.load(path_model3D, map_location=device)
        # gen_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint2['state_dict'].items()}
        # model3D.load_state_dict(gen_loaded_state)
        # model3D.eval()
        
        
        # path_model3D = './checkpoint/sch2_phase1_avec_weight_v2_' + country_name + '.ckpt' # ------ to use with phase 1 (avec_weight_v2)
        path_model3D = './checkpoint/sch2_phase1_avec_full_weight_' + country_name + '.ckpt'
        checkpoint2 = torch.load(path_model3D, map_location=device)
        gen_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint2['state_dict'].items()}
        model3D.load_state_dict(gen_loaded_state)
        model3D.eval()


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

                    label_pair = label_image_1[:] == label_image_2[:]  # label of pairs: 1 if the two images in the pair are
                    # of the same class, 0 if the images belong to two different classes
                    label_pair = label_pair.long()
                    latent_1, decoder_op_1 = model3D(img1)
                    latent_2, decoder_op_2 = model3D(img2)

                    class_op_1, y1o_softmax = model(latent_1)
                    class_op_2, y2o_softmax = model(latent_2)
                    
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
        with open("module_1_sch2_AUC" + ".csv", 'a') as file:
            writer = csv.writer(file)
            for myList in auc_list:
                file.write(str(myList) )
                file.write (',')
            file.write ('\n')
        
if __name__ == '__main__':
    main()

