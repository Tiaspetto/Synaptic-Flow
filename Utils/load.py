import torch
from data import Data
import numpy as np
from torchvision import datasets, transforms
import torch.optim as optim
from Models import mlp
from Models import lottery_vgg
from Models import lottery_resnet
from Models import tinyimagenet_vgg
from Models import tinyimagenet_resnet
from Models import imagenet_vgg
from Models import imagenet_resnet
from Pruners import pruners
from Models import sr_model
from Utils import custom_datasets
from pathlib import Path 
import os
def device(gpu):
    use_cuda = torch.cuda.is_available()
    return torch.device(("cuda:" + str(gpu)) if use_cuda else "cpu")

def dimension(dataset):
    if dataset == 'div2k':
        input_shape, output_shape = (3, 320, 180), (3, 1280, 720)
    return input_shape, output_shape

def get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)

def dataloader(dataset, batch_size, train, workers, length=None):
    # Dataset
    if dataset == 'mnist':
        mean, std = (0.1307,), (0.3081,)
        transform = get_transform(size=28, padding=0, mean=mean, std=std, preprocess=False)
        dataset = datasets.MNIST('Data', train=train, download=True, transform=transform)
    if dataset == 'cifar10':
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = datasets.CIFAR10('Data', train=train, download=True, transform=transform) 
    if dataset == 'cifar100':
        mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = datasets.CIFAR100('Data', train=train, download=True, transform=transform)
    if dataset == 'tiny-imagenet':
        mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
        transform = get_transform(size=64, padding=4, mean=mean, std=std, preprocess=train)
        dataset = custom_datasets.TINYIMAGENET('Data', train=train, download=True, transform=transform)
    if dataset == 'imagenet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2,1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        folder = 'Data/imagenet_raw/{}'.format('train' if train else 'val')
        dataset = datasets.ImageFolder(folder, transform=transform)
    if dataset == 'div2k':
        home = str(Path.home())
        dir_data = os.path.join(home, 'srprunner/dataset')
        dir_data_test = os.path.join(home, 'srprunner/dataset')
        data_train = 'DIV2K'
        data_test = 'Urban100'
        test_only = False
        batch_size = 8
        cpu = False
        n_GPUs = 1
        scale = 4
        test_every = 1000
        n_colors = 3
        rgb_range = 255
        patch_size = 192
        dataloader = Data(dir_data=dir_data, 
                          dir_data_test=dir_data_test, 
                          data_train=data_train, 
                          data_test=data_test, 
                          test_only=test_only, 
                          batch_size=batch_size,
                          cpu=cpu, 
                          n_GPUs=n_GPUs, 
                          scale=scale, 
                          test_every=test_every, 
                          n_colors=n_colors, 
                          rgb_range=rgb_range, 
                          patch_size=patch_size)
    else:
        # Dataloader
        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
        shuffle = train is True
        if length is not None:
            indices = torch.randperm(len(dataset))[:length]
            dataset = torch.utils.data.Subset(dataset, indices)

        dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle, 
                                                **kwargs)

    return dataloader

def model(model_architecture, model_class):
    default_models = {
        'fc' : mlp.fc,
        'conv' : mlp.conv,
    }
    lottery_models = {
        'vgg11' : lottery_vgg.vgg11,
        'vgg11-bn' : lottery_vgg.vgg11_bn,
        'vgg13' : lottery_vgg.vgg13,
        'vgg13-bn' : lottery_vgg.vgg13_bn,
        'vgg16' : lottery_vgg.vgg16,
        'vgg16-bn' : lottery_vgg.vgg16_bn,
        'vgg19' : lottery_vgg.vgg19,
        'vgg19-bn' : lottery_vgg.vgg19_bn,
        'resnet20': lottery_resnet.resnet20,
        'resnet32': lottery_resnet.resnet32,
        'resnet44': lottery_resnet.resnet44,
        'resnet56': lottery_resnet.resnet56,
        'resnet110': lottery_resnet.resnet110,
        'resnet1202': lottery_resnet.resnet1202,
        'wide-resnet20': lottery_resnet.wide_resnet20,
        'wide-resnet32': lottery_resnet.wide_resnet32,
        'wide-resnet44': lottery_resnet.wide_resnet44,
        'wide-resnet56': lottery_resnet.wide_resnet56,
        'wide-resnet110': lottery_resnet.wide_resnet110,
        'wide-resnet1202': lottery_resnet.wide_resnet1202
    }
    tinyimagenet_models = {
        'vgg11' : tinyimagenet_vgg.vgg11,
        'vgg11-bn' : tinyimagenet_vgg.vgg11_bn,
        'vgg13' : tinyimagenet_vgg.vgg13,
        'vgg13-bn' : tinyimagenet_vgg.vgg13_bn,
        'vgg16' : tinyimagenet_vgg.vgg16,
        'vgg16-bn' : tinyimagenet_vgg.vgg16_bn,
        'vgg19' : tinyimagenet_vgg.vgg19,
        'vgg19-bn' : tinyimagenet_vgg.vgg19_bn,
        'resnet18' : tinyimagenet_resnet.resnet18,
        'resnet34' : tinyimagenet_resnet.resnet34,
        'resnet50' : tinyimagenet_resnet.resnet50,
        'resnet101' : tinyimagenet_resnet.resnet101,
        'resnet152' : tinyimagenet_resnet.resnet152,
        'wide-resnet18' : tinyimagenet_resnet.wide_resnet18,
        'wide-resnet34' : tinyimagenet_resnet.wide_resnet34,
        'wide-resnet50' : tinyimagenet_resnet.wide_resnet50,
        'wide-resnet101' : tinyimagenet_resnet.wide_resnet101,
        'wide-resnet152' : tinyimagenet_resnet.wide_resnet152,
    }
    imagenet_models = {
        'vgg11' : imagenet_vgg.vgg11,
        'vgg11-bn' : imagenet_vgg.vgg11_bn,
        'vgg13' : imagenet_vgg.vgg13,
        'vgg13-bn' : imagenet_vgg.vgg13_bn,
        'vgg16' : imagenet_vgg.vgg16,
        'vgg16-bn' : imagenet_vgg.vgg16_bn,
        'vgg19' : imagenet_vgg.vgg19,
        'vgg19-bn' : imagenet_vgg.vgg19_bn,
        'resnet18' : imagenet_resnet.resnet18,
        'resnet34' : imagenet_resnet.resnet34,
        'resnet50' : imagenet_resnet.resnet50,
        'resnet101' : imagenet_resnet.resnet101,
        'resnet152' : imagenet_resnet.resnet152,
        'wide-resnet50' : imagenet_resnet.wide_resnet50_2,
        'wide-resnet101' : imagenet_resnet.wide_resnet101_2,
    }
    sr_models = {
        'rcan' : sr_model.rcan
    }

    models = {
        'default' : default_models,
        'lottery' : lottery_models,
        'tinyimagenet' : tinyimagenet_models,
        'imagenet' : imagenet_models,
        'srmodel' : sr_models
    }
    if model_class == 'imagenet':
        print("WARNING: ImageNet models do not implement `dense_classifier`.")
    return models[model_class][model_architecture]

def pruner(method):
    prune_methods = {
        'rand' : pruners.Rand,
        'mag' : pruners.Mag,
        'snip' : pruners.SNIP,
        'grasp': pruners.GraSP,
        'synflow' : pruners.SynFlow,
    }
    return prune_methods[method]

def optimizer(optimizer):
    optimizers = {
        'adam' : (optim.Adam, {}),
        'sgd' : (optim.SGD, {}),
        'momentum' : (optim.SGD, {'momentum' : 0.9, 'nesterov' : True}),
        'rms' : (optim.RMSprop, {})
    }
    return optimizers[optimizer]

