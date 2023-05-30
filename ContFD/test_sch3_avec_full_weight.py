import torch
import statistics
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from model import Generator, Discriminator 
from model_classify import Discriminator, Generator
import argparse
from data_5 import CsvDataset


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    args = parser.parse_args()
    batch_size = args.bs
    
    country_name = 'svk'
    
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
    model3D = Generator().to(device)  
    
    path = './checkpoint/sch3_avec_full_weight_' + country_name + '.ckpt' 
    checkpoint = torch.load(path, map_location=device)
    model_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(model_loaded_state)
    
    model3D_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint['model3D_state_dict'].items()}
    model3D.load_state_dict(model3D_loaded_state)
    model.eval()
    model3D.eval()
    
    ################
    get_corrects = 0.0
    accuracy_list = []
    
    testing_data_length = [15,30,45,60,75,90,105,120]
    for testing_length in testing_data_length:
        accuracy_init_list = []
        for epoch in range(3):
            get_corrects = 0.0
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
                
                get_corrects += torch.max(y1o_softmax, 1)[1] == label_image_1
                get_corrects += torch.max(y2o_softmax, 1)[1] == label_image_2
                          
            variable_acc = get_corrects.item() / (testing_length * 2)
            accuracy_init_list.append(variable_acc)
        # print('Accuracy@' + str(testing_length) + ': ', str(variable_acc))
        print('Accuracy@' + str(testing_length) + ': ', str(statistics.mean(accuracy_init_list)))
        accuracy_list.append(statistics.mean(accuracy_init_list))
    print('accuracy_list', accuracy_list)
if __name__ == '__main__':
    main()

