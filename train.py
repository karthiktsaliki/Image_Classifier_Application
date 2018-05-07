# Options to use
# Two options for using transfer learning on two different trained models
# python train.py --model vgg16 --use_gpu False --num_epochs 25 --alpha = 0.1 --lr 0.01 --hidden 100
# python train.py --model resnet18 --use_gpu False --num_epochs 25 --alpha = 0.1 --lr 0.01 --hidden 100

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image

data_dir = 'C:/Users/TsalikiK/Downloads/Kantar/Kantar_Python_Work/Notebooks/aipnd-project'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

def data_loaders():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test':transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: dsets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'valid','test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,shuffle=True, num_workers=4) for x in ['train', 'valid','test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}

    return dataloaders,dataset_sizes


def vgg16():
    model_conv = models.vgg16(pretrained='imagenet')
    for i, param in model_conv.named_parameters():
        param.requires_grad = False
    num_ftrs = model_conv.classifier[6].in_features
    features = list(model_conv.classifier.children())[:-1]
    features.extend([nn.Linear(num_ftrs, 102)])
    model_conv.classifier = nn.Sequential(*features)
    return model_conv

def resnet():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 102)
    return model_ft

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, use_gpu, num_epochs=25, alpha = 0.1):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if type(outputs) == tuple:
                    outputs, _ = outputs
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='vgg16', 
                        help='Model Name')
    parser.add_argument('--use_gpu', type=bool, default=False, 
                        help='Run with GPU')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Total No of Epocs')
    parser.add_argument('--hidden', type=int, default=100,
                        help='Total no of hidden units')
    parser.add_argument('--lr', type=int, default=0.01,
                        help='Learning Rate')

    # returns parsed argument collection
    return parser.parse_args()

def main():
    # Retrieving the command line arguments
    in_arg = get_input_args()
    dataloaders,dataset_sizes=data_loaders()
    # Traning with the selected model
    if in_arg.model='vgg16':
        model=vgg16()
    elif in_arg.model='resnet18':
        model=resnet()
    # Getting the hyperparametes and use_gpu
    print("[Using CrossEntropyLoss...]")
    criterion = nn.CrossEntropyLoss()
    print("[Using small learning rate with momentum...]")
    optimizer_conv = optim.SGD(list(filter(lambda p: p.requires_grad, model_conv.parameters())), lr=in_arg.lr, momentum=0.9)
    print("[Creating Learning rate scheduler...]")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    print("[Training the model begun ....]")
    model_ft = train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler,in_arg.use_gpu, in_arg.num_epochs)
    # Saving the model
    print("[Saving the model and optimizer ....]")
    model_ft.class_to_idx = image_datasets['train'].class_to_idx
    torch.save(model_ft,'flowers_transfer_command.pt')
    torch.save(optimizer_conv.state_dict(), 'optimizers_command.pt')
        