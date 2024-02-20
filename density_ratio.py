import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import yaml
from density_NN import DensityRatioEstNet
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
if not os.path.exists(os.path.join(os.getcwd(),'MNIST_obj')):
    os.mkdir(os.path.join(os.getcwd(), 'MNIST_obj'))
if not os.path.exists(os.path.join(os.getcwd(),'MNIST_ref')):
    os.mkdir(os.path.join(os.getcwd(), 'MNIST_ref'))

config_file_path = os.path.join(os.getcwd(), "config.yml") 
with open(config_file_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

noise_std = config["data"]["noise_std"]
image_size = config["data"]["image_size"]
ngf = config["model"]["ngf_d"]
channels = config["data"]["channels"]
obj_size = config["data"]["obj_size"] # Number of objective samples in training
target_label = config["data"]["target_label"]
batch_size_d = config["training"]["batch_size"]
num_workers_d = config["training"]["num_workers"]
num_iter = config["training"]["num_iter_d"]

add_gaussian_noise = transforms.Lambda(lambda x: x + torch.randn_like(x) * noise_std)
tran_transform_obj = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    add_gaussian_noise
])
tran_transform_ref = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])


dataset_obj = MNIST(os.path.join(os.getcwd(),'MNIST_obj'), train=True, download=True,
                transform=tran_transform_obj)
dataset_ref = MNIST(os.path.join(os.getcwd(),'MNIST_ref'), train=True, download=True,
                transform=tran_transform_ref)


filtered_indices_obj = [idx for idx in range(len(dataset_obj)) if dataset_obj.targets[idx] == target_label][:obj_size]
filtered_indices_ref = [idx for idx in range(len(dataset_ref)) if dataset_ref.targets[idx] != target_label]

dataset_obj = torch.utils.data.Subset(dataset_obj, filtered_indices_obj)
dataset_ref = torch.utils.data.Subset(dataset_ref, filtered_indices_ref)

desired_dataset_size = len(dataset_ref)
repetitions = desired_dataset_size // len(dataset_obj)
remaining_samples = desired_dataset_size % len(dataset_obj) 
extended_dataset = torch.utils.data.ConcatDataset([dataset_obj] * repetitions + [Subset(dataset_obj, list(range(remaining_samples)))])
dataset_obj = extended_dataset

D = DensityRatioEstNet(ngf,image_size, channels).to(device)
optimizerD = optim.Adam(
    D.parameters(), 
    lr=config["optim"]["lr_d"], 
    weight_decay=config["optim"]["weight_decay"], 
    betas=(config["optim"]["beta1"], config["optim"]["beta2"])
)
D.train()
train_loader_ref = data.DataLoader(
    dataset_ref,
    batch_size=batch_size_d,
    shuffle=True,
    num_workers=num_workers_d
)
train_loader_obj = data.DataLoader(
    dataset_obj,
    batch_size=batch_size_d,
    shuffle=True,
    num_workers=num_workers_d
)


start_epoch, step = 0, 0
for epoch in range(start_epoch,  num_iter):
    print(epoch)
    for (obj_data, ref_data) in zip(train_loader_obj,train_loader_ref):
        (X, y) = obj_data
        (Z, r) = ref_data   
        step += 1
        X = X.to(device)
        Z = Z.to(device)
        real_score = D(Z)	
        fake_score = D(X)
        optimizerD.zero_grad()
        loss_d_real = torch.log(1 + torch.exp(-real_score)).mean()
        loss_d_fake = torch.log(1 + torch.exp(fake_score)).mean()
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizerD.step()
        if not step % 100:
                print(loss_d)

state = {'D': D.state_dict()}

if not os.path.exists(os.path.join(os.getcwd(), 'Model')):
    os.mkdir(os.path.join(os.getcwd(), 'Model'))



torch.save(state, os.path.join(os.getcwd(), "Model","checkpoint_density_estimation.pth"))

