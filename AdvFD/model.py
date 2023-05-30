import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F



# the only difference of this code with the one in "model.py" is that here the final ouput has 3 channels
class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.convA(torch.cat([up_x, concat_with], dim=1))))


class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width=1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features // 1 + 256, output_features=features // 2)
        self.up2 = UpSample(skip_input=features // 2 + 128, output_features=features // 4)
        self.up3 = UpSample(skip_input=features // 4 + 64, output_features=features // 8)
        self.up4 = UpSample(skip_input=features // 8 + 64, output_features=features // 16)
        # self.up4 = UpSample(skip_input=features // 8 + 3, output_features=features // 16)

        # self.conv3 = nn.Conv2d(features // 16, 3, kernel_size=3, stride=1, padding=1)  # here, I have changed the
        # # number of channels from 1 to 3
        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block_int, x_block0, x_block1, x_block2, x_block3, x_block4 = features[0], features[3], features[4], features[6], features[8], features[
            12]

        x_d0 = self.conv2(F.relu(x_block4))
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = models.densenet169(pretrained=False)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 240 * 240, 512), # 300 * 300
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64))

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1):
        # output1 = self.forward_once(input1)
        # output2 = self.forward_once(input2)

        input1 = input1.repeat(1, 3, 1, 1)

        encoder_op_1 = self.encoder(input1)

        decoder_op_1 = self.decoder(encoder_op_1)


        return encoder_op_1[-1], decoder_op_1


# class Discriminator1(nn.Module):
#     def __init__(self):
#         super(Discriminator1, self).__init__()
#         # Fully connected layer
#         self.fc1 = torch.nn.Linear(1664*7*7, 2**11)   
#         self.fc2 = torch.nn.Linear(2**11, 2**9)
#         self.fc3 = torch.nn.Linear(2**9, 2**5)
#         self.fc4 = torch.nn.Linear(2**5, 2**1)      
#         #self.fc5 = torch.nn.Linear(2**1, 2**0) 
#         #self.fc6 = nn.Sigmoid()
    
#     def forward(self, x):
#         x = x.view(-1, 1664*7*7)
#         # x = torch.transpose(x, 0, 1)
#         # x = x.unsqueeze(0)  # B, C, T, H, W
#         # FC-1, then perform ReLU non-linearity
#         x = torch.nn.functional.relu(self.fc1(x))
#         # FC-2, then perform ReLU non-linearity
#         x = torch.nn.functional.relu(self.fc2(x))
#         # FC-3 then perform ReLU non-linearity
#         x = torch.nn.functional.relu(self.fc3(x)) 
#         # FC-4 then perform ReLU non-linearity
#         x = torch.nn.functional.relu(self.fc4(x))
#         y1o_softmax = F.softmax(x, dim=1)
#         # FC-5 then perform ReLU non-linearity
#         #x = torch.nn.functional.relu(self.fc5(x))
#         #x = self.fc6(x)
#         return x, y1o_softmax



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Fully connected layer
        self.fc1 = nn.Linear(1664*7*7, 1024)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = nn.Linear(1024, 128)       # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = nn.Linear(128, 32)        # convert matrix with 84 features to a matrix of 10 features (columns)
        self.fc4 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = x.view(-1, 1664*7*7)
        # FC-1, then perform ReLU non-linearity
        x = nn.functional.relu(self.fc1(x))
        # x = nn.BatchNorm2d(x)
        # FC-2, then perform ReLU non-linearity
        x = nn.functional.relu(self.fc2(x))
        # FC-3 then perform ReLU non-linearity
        x = nn.functional.relu(self.fc3(x))
        # FC-4
        x = self.fc4(x)
        y1o_softmax = F.softmax(x, dim=1)
        return x, y1o_softmax



# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         # Fully connected layer
#         self.fc1 = nn.Linear(1664*7*7, 1024)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
#         self.fc2 = nn.Linear(1024, 128)       # convert matrix with 120 features to a matrix of 84 features (columns)
#         self.fc3 = nn.Linear(128, 32)        # convert matrix with 84 features to a matrix of 10 features (columns)
#         self.fc4 = nn.Linear(32, 2)

#         self.bn1 = nn.BatchNorm1d(1024)  
#         self.bn2 = nn.BatchNorm1d(128)       
#         self.bn3 = nn.BatchNorm1d(32)    


    
#     def forward(self, x):
#         x = x.view(-1, 1664*7*7)
#         # FC-1, then perform ReLU non-linearity
#         x = self.bn1(F.relu(self.fc1(x)))
#         # FC-2, then perform ReLU non-linearity
#         x = self.bn2(F.relu(self.fc2(x)))
#         # FC-3 then perform ReLU non-linearity
#         x = self.bn3(F.relu(self.fc3(x)))
#         # FC-4
#         x = self.fc4(x)
#         y1o_softmax = F.softmax(x, dim=1)
#         return x, y1o_softmax