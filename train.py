import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import time
import copy
import argparse

# Defining Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help="Path to dataset")
parser.add_argument('--gpu', action='store_true', help="Use GPU if available")
parser.add_argument('--epochs', type=int, help="Number of epochs")
parser.add_argument('--arch', type=str, help="Model Architechture")
parser.add_argument('--learning_rate', type=float, help="Learning rate")
parser.add_argument('--hidden_units', type=int, help="Number of hidden units")
parser.add_argument('--checkpoint', type=str, help="Save trained model checkpoint to file")

args, _ = parser.parse_known_args()

def load_model(arch='vgg19', output_labels=102, hidden_units=4096):
    if arch=='vgg19':
        model = models.vgg19(pretrained=True)
    elif arch=='alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("There is a problem with the chosen architecture")
        exit
    
    # Free arch parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # get features removing just the last layer
    arch_features = list(model.classifier.children())[:-1]
    # print(arch_features)

    # number of filters in the layer
    num_filters = model.classifier[len(arch_features)].in_features

    # extend exisitng features
    arch_features.extend([
        nn.Dropout()
        ,nn.Linear(num_filters, hidden_units)
        ,nn.ReLU(True)
        ,nn.Dropout()
        ,nn.Linear(hidden_units, hidden_units)
        ,nn.ReLU(True)
        ,nn.Linear(hidden_units, output_labels)
        # Skipping log softmax as per https://github.com/pytorch/vision/issues/432#issuecomment-368330817
    ])

    model.classifier = nn.Sequential(*arch_features)

    return model

# train a model
def train_model(image_datasets, arch='vgg19', hidden_units=4096, epochs=15, learning_rate=0.001, gpu=False, checkpoint=''):
    # get arguments from command line
    if args.arch:
        arch = args.arch
    if args.hidden_units:
        hidden_units = args.hidden_units
    if args.epochs:
        epochs = args.epochs
    if args.learning_rate:
        learning_rate = args.learning_rate
    if args.gpu:
        gpu = args.gpu
    if args.checkpoint:
        checkpoint = args.checkpoint
    
    dataloaders = {
        x: data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=2)
        for x in list(image_datasets.keys())
    }

    # calculate dataset sizes
    dataset_sizes = {
        x: len(dataloaders[x].dataset)
        for x in list(image_datasets.keys())
    }

    print("Network architecture name", arch)
    print('No. of hidden units', hidden_units)
    print('Number of epochs', epochs)
    print('Learning Rate', learning_rate)

    # load the model
    num_labels = len(image_datasets['train'].classes)
    model = load_model(arch=arch, output_labels=num_labels, hidden_units=hidden_units)

    # use gpu if available
    if gpu and torch.cuda.is_available():
        print('GPU is available for training')
        device = torch.device("cuda:0")
        model.cuda()
    else:
        print('Using cpu for training')
        device = torch.device("cpu")
    
    # defining criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-'*10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            # train
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    model.class_to_idx = image_datasets['train'].class_to_idx

    if checkpoint:
        print('Saving checkpoint: ', checkpoint)
        checkpoint_dict = {
            'arch': arch,
            'class_to_idx': model.class_to_idx,
            'state_dict': model.state_dict(),
            'hidden_units': hidden_units
        }

        torch.save(checkpoint_dict, checkpoint)
    
    return model

if args.data_dir:  
    # Default transforms for the training, validation, and testing sets
    # print('hello')
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ])
    }
    
    # Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(root=args.data_dir + '/' + x, transform=data_transforms[x])
        for x in list(data_transforms.keys())
    }
        
    train_model(image_datasets)