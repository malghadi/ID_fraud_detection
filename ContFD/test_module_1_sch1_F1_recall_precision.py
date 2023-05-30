import torch
import statistics
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Generator as Model3D
import argparse
from data_5 import CsvDataset
import csv
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
        print(country_name)
        F_score_list = [] 
        tpr_list = []
        fpr_list = []
        precision_list = []
        recall_list = []   

        test_csv_dir = './data/testing/testing_' + country_name + '.csv' 
        testing_data_length = [15,30,45,60,75,90,105,120]
        for testing_length in testing_data_length:
            batch_size = testing_length
            print('batch_size', batch_size)

            # Load data
            transform1 = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])
            transform2 = transforms.Compose([transforms.Resize((120, 120)), transforms.ToTensor()])
            testing_dataset = CsvDataset(csv_file=test_csv_dir, transform1=transform1, transform2=transform2, should_invert=False)
            test_loader = DataLoader(testing_dataset, shuffle=True, num_workers=0 * torch.cuda.device_count(), batch_size=batch_size, pin_memory=True)
            print('test_loader', len(test_loader))

            # is_use_cuda = torch.cuda.is_available()
            # device = torch.device("cuda" if is_use_cuda else "cpu")
            device = torch.device("cpu")

            #################
            model3D = Model3D().to(device)    
            # path_model3D = './checkpoint/siamese_net/module_1_sch1_' + country_name + '.ckpt'
            path_model3D = './checkpoint/sch1_avec_full_weight_' + country_name + '.ckpt'
            checkpoint = torch.load(path_model3D, map_location=device)
            gen_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            model3D.load_state_dict(gen_loaded_state)
            model3D.eval()
            
            #--------------------------------------------
            F_score_in = [] 
            tpr_in = [] 
            fpr_in = []
            precision_in = []
            recall_in =[]   
            for epoch in range(3):
                for data in test_loader:
                    # Prepare sample and target
                    img1, img1_half, img2, img2_half, label_1, label_2 = data  # label_0 is the label of image_0 and label_1 is for image_1
                    img1, img1_half, img2, img2_half, label_1, label_2 = img1.to(device), img1_half.to(device), img2.to(
                        device), img2_half.to(device), label_1.to(device), label_2.to(device)

                    # latent_1, latent_2, class_op_1, y1o_softmax, class_op_2, y2o_softmax, decoder_op_1, decoder_op_2 = model3D(img1, img2)
                    latent_1, class_op_1, y1o_softmax, decoder_op_1 = model3D(img1)
                    latent_2, class_op_2, y2o_softmax, decoder_op_2 = model3D(img2)

                    # calculating true positive rate (TPR) and false negative rate (FNR) -------------------- !! Remember !! TPR + FNR = 1
                    # https://subscription.packtpub.com/book/data/9781800564480/2/ch02lvl1sec13/model-performance-metrics-for-binary-classification
                    
                    #--------- for image_1
                    P_1 = sum(label_1) #  calculate the number of positive samples for image_1
                    TP_1 = sum((label_1==1) & (torch.max(y1o_softmax, 1)[1]==1))
                    # TPR = TP/P
                    FN_1 = sum((label_1==1) & (torch.max(y1o_softmax, 1)[1]==0))
                    # FNR = FN/P
                    #--------- for image_2
                    P_2 = sum(label_2) #  calculate the number of positive samples for image_2
                    TP_2 = sum((label_2==1) & (torch.max(y2o_softmax, 1)[1]==1))
                    FN_2 = sum((label_2==1) & (torch.max(y2o_softmax, 1)[1]==0))
                    TPR = (TP_1+TP_2)/(P_1+P_2)
                    FNR = (FN_1+FN_2)/(P_1+P_2)

                    # calculating true negative rate (TNR) and false positive rate (FPR)---------------------
                    #--------- for image_1
                    N_1 = sum(label_1 == 0) #  calculate the number of negative samples for image_1
                    TN_1 = sum((label_1 == 0) & (torch.max(y1o_softmax, 1)[1] == 0))
                    # TNR = TN/N
                    FP_1 = sum((label_1==0) & (torch.max(y1o_softmax, 1)[1]==1))
                    # FPR = FP/N
                    #--------- for image_2
                    N_2 = sum(label_2 == 0) #  calculate the number of negative samples for image_2
                    TN_2 = sum((label_2 == 0) & (torch.max(y2o_softmax, 1)[1] == 0))
                    FP_2 = sum((label_2==0) & (torch.max(y2o_softmax, 1)[1]==1))
                    TNR = (TN_1+TN_2)/(N_1+N_2)
                    FPR = (FP_1+FP_2)/(N_1+N_2)
                    
                    F_score = TPR / (TPR + 0.5 * (FPR + FNR))
                    precision = TPR / (TPR + FPR)
                    recall = TPR / (TPR+FNR)
                    F_score_in.append(F_score.item())
                    precision_in.append(precision.item())
                    recall_in.append(recall.item())
                    tpr_in.append(TPR.item())
                    fpr_in.append(FPR.item())
                    break            
            F_score_list.append(statistics.mean(F_score_in))
            precision_list.append(statistics.mean(precision_in))
            recall_list.append(statistics.mean(recall_in))
            tpr_list.append(statistics.mean(tpr_in))
            fpr_list.append(statistics.mean(fpr_in))
            print('F_score_list', F_score_list)
        # print('precision_list', precision_list)
        # print('tpr_list', tpr_list)
        # print('fpr_list', fpr_list)  
        
        if not os.path.isdir('./results'):
                os.makedirs('./results')
        with open("results/F1_scors.csv", 'a') as file:
            writer = csv.writer(file)
            for myList in F_score_list:
                file.write(str(myList) )
                file.write (',')
            file.write ('\n')

        with open("results/precision.csv", 'a') as file:
            writer = csv.writer(file)
            for myList in precision_list:
                file.write(str(myList) )
                file.write (',')
            file.write ('\n')

        with open("results/recall.csv", 'a') as file:
            writer = csv.writer(file)
            for myList in recall_list:
                file.write(str(myList) )
                file.write (',')
            file.write ('\n')

if __name__ == '__main__':
    main()
