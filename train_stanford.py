"""
Chuhang Zou
07.2019

Code Revised from:

Finetuning Torchvision Models
=============================

**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__

"""

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
import time
import copy
print("PyTorch Version: ",torch.__version__)

from torch.utils import data
from model import *
from data_generator import *

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
model_path = './model/resnet34_stanford0.pth'

train_datapath = './data/train_stanford/'
val_datapath = './data/val_stanford/'

# Pre-trained models to choose from [resnet18, resnet34, resnet50]
#model_name = "resnet18"
model_name = "resnet34"
#model_name = "resnet50"

# if load pretrained model
Flag_loadweights = True
weight_path = './model/resnet34_stanford.pth'

# Number of classes in the dataset
num_classes = 1024

# Batch size for training (change depending on how much memory you have)
batch_size = 4

# Number of epochs to train for 
num_epochs = 100000
steps_per_epoch = 20


# Model Training and Validation Code
def train_model(model, train_generator, val_generator, optimizer, criterion, criterion2, steps=100, num_epochs=25):
    since = time.time()

    val_acc_history = []
    
    best_acc = np.Inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloaders = train_generator
            else:
                model.eval()   # Set model to evaluate mode
                dataloaders = val_generator

            loss_sum = 0.0
            step = 0

            # Iterate over data.
            for input in dataloaders:
                
                inputs = input[0]
                labels = input[1]
                labels2 = input[2]

                # gpu mode
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels2 = labels2.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs, outputs2 = model(inputs)
                    
                    # loss
                    mask = (labels>0).float()
                    mask = mask+0.2*(1.0-mask)
                    mask2 = (labels2>0).float()
                    mask2 = mask2+0.2*(1.0-mask2)
                    
                    loss = criterion(outputs*mask, labels*mask) + criterion2(outputs2*mask2, labels2*mask2)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()

                # clear cache
                torch.cuda.empty_cache()
                del inputs, outputs, outputs2, mask, mask2, labels, labels2
                #del inputs, outputs, mask, labels

                # statistics
                loss_sum += loss.item()/steps #*inputs.size(0)/steps
                
                # clear
                del loss

                # Break after 'steps' steps
                if step==steps-1:
                    break
                step += 1
            
            print('{} Loss: {:.6f}'.format(phase, loss_sum))

            # deep copy the model
            if phase == 'val' and loss_sum < best_acc:
                val_acc_history.append(loss_sum)
                best_acc = loss_sum
                best_model_wts = copy.deepcopy(model.state_dict())
                # save model
                torch.save(best_model_wts, model_path)
                print("Model saved ...")
                del best_model_wts

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:6f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


print("Load Models...")
# Define the encoder
encoder = initialize_encoder(model_name, num_classes,use_pretrained=True)

# Full model
model_ft = SegNet(encoder, num_classes)

# Model initialization
set_parameter_requires_grad(model_ft)
# Print the model we just instantiated
#print(model_ft) 

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# if load weights
if Flag_loadweights:

    pretrained_dict = torch.load(weight_path)
    model_dict = model_ft.state_dict()
    #pretrained_dict['resnet.conv1.weight'] = torch.cat((pretrained_dict['resnet.conv1.weight'], model_dict['resnet.conv1.weight'][:,3:,:,:]), 1)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_ft.load_state_dict(model_dict)

# update header
print("Params to learn:")
for name,param in model_ft.named_parameters():
    if 'bn' in name:
    # tune hearder only
#    if name != 'conv11.bias' and name != 'conv11.weight' and name != 'conv22.weight' and name != 'conv22.bias' and name != 'conv33.bias' and name != 'conv33.weight' and name != 'conv44.bias' and name != 'conv44.weight' and name != 'conv55.bias' and name != 'conv55.weight':
        param.requires_grad = False
    if param.requires_grad == True:
        print("\t",name)

# Gather the parameters to be optimized/updated in this run.
params_to_update = [param for name, param in model_ft.named_parameters() if param.requires_grad]

# Create the Optimizer
optimizer_ft = optim.Adam(params_to_update, lr = 1e-4, eps = 1e-6)  

# Setup the loss
criterion = nn.BCELoss()
criterion2 = nn.BCELoss()
# Load Data
print("Initializing Datasets and Dataloaders...")
train_set = ShapeNetDataset(train_datapath, 'train', transform=True)
train_generator = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_set = ShapeNetDataset(val_datapath, 'val', transform=True)
val_generator = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)

# Train and evaluate
model_ft, hist = train_model(model_ft, train_generator, val_generator, optimizer_ft, criterion, criterion2, steps_per_epoch, num_epochs=num_epochs)

print('training done')
