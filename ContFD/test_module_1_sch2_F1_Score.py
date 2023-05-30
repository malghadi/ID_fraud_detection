import torch
import statistics
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model_classify import Discriminator, Generator as Model3D
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

        
        path_model3D = './checkpoint/sch2_phase1_avec_full_weight_' + country_name + '.ckpt'
        checkpoint2 = torch.load(path_model3D, map_location=device)
        gen_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint2['state_dict'].items()}
        model3D.load_state_dict(gen_loaded_state)
        model3D.eval()


        ################
        cnt_TAR = 0.0
        cnt_FRR = 0.0
        cnt_FAR = 0.0
        cnt_true = 0.0
        cnt_false = 0.0 
        
        testing_data_length = [15,30,45,60,75,90,105,120]
        for testing_length in testing_data_length:
            F_scores_init_list = []
            for epoch in range(1):
                F_scores = []
                for i, data in enumerate(test_loader):
                    # print(i)
                    if i == testing_length - 1:
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
                    
                    if label_image_2 == 1:
                        cnt_true += 1
                    else:
                        cnt_false += 1

                    if (label_image_2 == 1 and torch.max(y2o_softmax, 1)[
                        1] == label_image_2):  # truth label is 1 and detected as 1
                        cnt_TAR += 1
                    if (label_image_2 == 1 and torch.max(y2o_softmax, 1)[
                        1] != label_image_2):  # truth label is 1 and detected as 0
                        cnt_FRR += 1  # false rejection rate (false negative)
                    if (label_image_2 == 0 and torch.max(y2o_softmax, 1)[
                        1] != label_image_2):  # truth label is 0 and detected as 1
                        cnt_FAR += 1  # false acceptance rate  (false positive)

                    if label_image_2 == 0:
                        cnt_true += 1
                    else:
                        cnt_false += 1

                    if (label_image_2 == 0 and torch.max(y2o_softmax, 1)[
                        1] == label_image_2):  # truth label is 1 and detected as 1
                        cnt_TAR += 1
                    if (label_image_2 == 0 and torch.max(y2o_softmax, 1)[
                        1] != label_image_2):  # truth label is 1 and detected as 0
                        cnt_FRR += 1
                    if (label_image_2 == 1 and torch.max(y2o_softmax, 1)[1] != label_image_2):
                        cnt_FAR += 1

                TAR = cnt_TAR / cnt_true
                FRR = cnt_FRR / cnt_true
                FAR = cnt_FAR / cnt_false
                F_score = TAR / (TAR + 0.5 * (FAR + FRR))
                F_scores.append(F_score)
            F_scores_init_list.append(statistics.mean(F_scores))
            # print('F_scores_init_list:', testing_length, F_scores_init_list)

            F_score_list.extend(F_scores_init_list)
        print('F_score_list', country_name, F_score_list)
        if not os.path.isdir('./results'):
                os.makedirs('./results')
        with open("results/F_score_list" + "_model1_sch2_.csv", 'a') as file:
            writer = csv.writer(file)
            for myList in F_score_list:
                # writer.writerow([list])
                # line = '\t'.join([myVariable] + myList)
                file.write(str(myList))
                file.write (',')
            file.write ('\n')
if __name__ == '__main__':
    main()

