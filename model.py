
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import time
import resnet_seg_ae as resnet

# Initialize and Reshape the Encoders
def initialize_encoder(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet18":
        """ Resnet18
        """
        #model_ft = models.resnet18(pretrained=use_pretrained)
        model_ft = resnet.resnet18(pretrained=use_pretrained, num_classes=1000)
        #set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet34":
        """ Resnet34
        """
        #model_ft = models.resnet34(pretrained=use_pretrained)
        model_ft = resnet.resnet34(pretrained=use_pretrained, num_classes=1000)
        #set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet50":
        """ Resnet50
        """
        #model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft = resnet.resnet50(pretrained=use_pretrained, num_classes=1000)
        #set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet101":
        """ Resnet101
        """
        #model_ft = models.resnet101(pretrained=use_pretrained)
        model_ft = resnet.resnet101(pretrained=use_pretrained, num_classes=1000)
        #set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

# full model
class SegNet(nn.Module):
    def __init__(self, encoder, num_classes):
        super(SegNet, self).__init__()
        self.resnet = encoder
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # resnet50
#        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1)
#        self.conv2 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
#        self.conv3 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
#        self.conv4 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
#        self.conv5 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
#        #self.conv5 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

#        self.conv11 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1)
#        self.conv22 = nn.Conv2d(2048+1024, 512, kernel_size=3, stride=1, padding=1)
#        self.conv33 = nn.Conv2d(1024+512, 256, kernel_size=3, stride=1, padding=1)
#        self.conv44 = nn.Conv2d(512+256, 64, kernel_size=3, stride=1, padding=1)
#        self.conv55 = nn.Conv2d(128+64, 1, kernel_size=3, stride=1, padding=1)

        # resnet18/34
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        #self.conv5 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
        
        self.conv11 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(512+256, 128, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(256+128, 64, kernel_size=3, stride=1, padding=1)
        self.conv44 = nn.Conv2d(128+64, 64, kernel_size=3, stride=1, padding=1)
        self.conv55 = nn.Conv2d(128+64, 1, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        x4, x3, x2, x1, x0 = self.resnet(images)
        
        # decoder
        # lay
        x3_ = self.relu(self.conv1(self.upsample(x4)))
        x3_c = torch.cat((x3_, x3),1)
        x2_ = self.relu(self.conv2(self.upsample(x3_c)))
        x2_c = torch.cat((x2_, x2),1)
        x1_ = self.relu(self.conv3(self.upsample(x2_c)))
        x1_c = torch.cat((x1_, x1),1)
        x0_ = self.relu(self.conv4(self.upsample(x1_c)))
        x0_c = torch.cat((x0_, x0),1)
        x0_cs = self.sigmoid(self.conv5(self.upsample(x0_c)))
        
        # cor
        x = self.relu(self.conv11(self.upsample(x4)))
        x = torch.cat((x, x3_c),1)
        x = self.relu(self.conv22(self.upsample(x)))
        x = torch.cat((x, x2_c),1)
        x = self.relu(self.conv33(self.upsample(x)))
        x = torch.cat((x, x1_c),1)
        x = self.relu(self.conv44(self.upsample(x)))
        x = torch.cat((x, x0_c),1)
        x = self.sigmoid(self.conv55(self.upsample(x)))
        
        return x0_cs, x

# Set Model Parameters, requires_grad attribute
def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = True
