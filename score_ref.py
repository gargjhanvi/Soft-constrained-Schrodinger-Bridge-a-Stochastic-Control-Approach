import torch
import numpy as np
import os
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import yaml
from density_NN import DensityRatioEstNet
from score_NN import CondRefineNetDilated
from losses import dsm_score_estimation_ref
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, help="Regularization parameter in Soft constrained Schrodinger Bridge problem")
args = parser.parse_args()
beta = args.beta

config_file_path = os.path.join(os.getcwd(), "config.yml") 
with open(config_file_path, 'r') as config_file:
    config = yaml.safe_load(config_file)


if not os.path.exists(os.path.join(os.getcwd(),'MNIST_obj')):
    os.mkdir(os.path.join(os.getcwd(), 'MNIST_obj'))
if not os.path.exists(os.path.join(os.getcwd(),'MNIST_ref')):
    os.mkdir(os.path.join(os.getcwd(), 'MNIST_ref'))

if not os.path.exists(os.path.join(os.getcwd(),'Model',"score_ref")):
    os.mkdir(os.path.join(os.getcwd(),'Model', 'score_ref'))
   
if not os.path.exists(os.path.join(os.getcwd(),'Model',"score_obj")):
    os.mkdir(os.path.join(os.getcwd(),'Model', 'score_obj'))
   

noise_std = config["data"]["noise_std"]
add_gaussian_noise = transforms.Lambda(lambda x: x + torch.randn_like(x) * noise_std)
image_size = config["data"]["image_size"]
target_label = config["data"]["target_label"]
batch = config["training"]["batch_size"]
num_work =  config["training"]["num_workers"]
device = config["training"]["device"]
ngf = config["model"]["ngf"]
channels = config["data"]["channels"]
sigma_begin = config["model"]["sigma_begin"]
sigma_end = config["model"]["sigma_end"]
num_classes =config["model"]["num_classes"]
n_epochs = config["training"]["n_epochs"]
n_iters = config["training"]["n_iters"]
snapshot_freq = config["training"]["snapshot_freq"]

tran_transform_obj = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    add_gaussian_noise
])
tran_transform_ref = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

dataset_obj = MNIST(os.path.join(os.getcwd(), 'MNIST_obj'), train=True, download=True,
                transform=tran_transform_obj)
dataset_ref = MNIST(os.path.join(os.getcwd(),'MNIST_ref'), train=True, download=True,
                transform=tran_transform_ref)
filtered_indices_obj = [idx for idx in range(len(dataset_obj)) if dataset_obj.targets[idx] == target_label]
filtered_indices_ref = [idx for idx in range(len(dataset_ref)) if dataset_ref.targets[idx] != target_label]
filtered_indices_objective = filtered_indices_obj[:config["data"]["obj_size"]]
dataset_obj = torch.utils.data.Subset(dataset_obj, filtered_indices_objective)
dataset_ref = torch.utils.data.Subset(dataset_ref, filtered_indices_ref)
desired_dataset_size =  len(dataset_ref)
repetitions = desired_dataset_size // len(dataset_obj) 
remaining_samples = desired_dataset_size % len(dataset_obj) 
extended_dataset = torch.utils.data.ConcatDataset([dataset_obj] * repetitions + [Subset(dataset_obj, list(range(remaining_samples)))])

dataset_obj = extended_dataset 
dataloader_obj = DataLoader(dataset_obj, batch_size=batch, shuffle=True, num_workers=num_work)
dataloader_ref = DataLoader(dataset_ref, batch_size=batch, shuffle=True, num_workers=num_work)
input_dim = image_size**2 
score = CondRefineNetDilated(config).to(device)
score = torch.nn.DataParallel(score)
D_path = torch.load(os.path.join(os.getcwd(), "Model", "checkpoint_density_estimation.pth"), map_location=torch.device(device))
model_path_1 = D_path['D']
D =  DensityRatioEstNet(batch, image_size, channels).to(device)
D.load_state_dict(model_path_1)
optimizer = optim.Adam(score.parameters(), lr=config["optim"]["lr_s"], weight_decay=config["optim"]["weight_decay"],betas=(config["optim"]["beta1"], config["optim"]["beta2"]), amsgrad=config["optim"]["amsgrad"]) 
sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),num_classes))).float().to(device)

step = 0
def train_ref(beta):
    step = 0
    for epoch in range(n_epochs):
        for (obj_data, ref_data) in zip(dataloader_obj,dataloader_ref):
            (X, y) = obj_data 
            (Z, r) = ref_data   
            step += 1
            score.train()
            X = X.to(device)
            X = X / 256. * 255. + torch.rand_like(X) / 256.
            labels = torch.randint(0, len(sigmas), (X.shape[0],), device=device)
            Z = Z.to(device)
            Z = Z / 256. * 255. + torch.rand_like(Z) / 256.
            labels = torch.randint(0, len(sigmas), (Z.shape[0],), device=device)
            lossb = dsm_score_estimation_ref(score,D, beta, Z, labels, sigmas) 
            loss = lossb
            optimizer.zero_grad()
            loss.backward()
            print(step)
            print(loss)
            optimizer.step()
            if step >= n_iters:
                return 0
            if step % 100 == 0:
                score.eval()
            if step % snapshot_freq== 0:
                states = [
                    score.state_dict(),
                    optimizer.state_dict(),
                ]
                torch.save(states, os.path.join(os.getcwd(), "Model","score_ref", 'checkpoint_{}.pth'.format(step)))
                torch.save(states, os.path.join(os.getcwd(), "Model","score_ref", 'checkpoint.pth'))


train_ref(beta)

