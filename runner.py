import numpy as np
import tqdm
import torch
import os
from torchvision.utils import save_image, make_grid
from PIL import Image
import yaml
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import argparse
from density_NN import DensityRatioEstNet
from score_NN import CondRefineNetDilated
import math

parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, help="Regularization parameter in Soft constrained Schrodinger Bridge problem")
args = parser.parse_args()
beta = args.beta

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
a = 0.5
if not os.path.exists(os.path.join(os.getcwd(),'Samples')):
    os.mkdir('Samples')
config_file_path = os.path.join(os.getcwd(), "config.yml") 
with open(config_file_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

states1 = torch.load(os.path.join(os.getcwd(), 'Model','score_ref','checkpoint.pth'),map_location = torch.device(device)) 
score1 = CondRefineNetDilated(config).to(device)
score1 = torch.nn.DataParallel(score1)
score1.load_state_dict(states1[0])
score1.eval()


states2 = torch.load(os.path.join(os.getcwd(), 'Model','score_obj','checkpoint_15000.pth'),map_location = torch.device(device)) 
score2 = CondRefineNetDilated(config).to(device)
score2 = torch.nn.DataParallel(score2)
score2.load_state_dict(states2[0])
score2.eval()
sigmas = torch.tensor(np.exp(np.linspace(np.log(config["model"]["sigma_begin"]), np.log(config["model"]["sigma_end"]),config["model"]["num_classes"]))).float().to(device)
def SSB(x_mod, score1, score2, beta,sigmas, n_steps_each=1000):
    k = n_steps_each/config["model"]["num_classes"]
    c = torch.cat([i * torch.ones(int(k)).to(device) for i in range(config["model"]["num_classes"])])
    sig = torch.cat([torch.ones(int(k)).to(device) * sigma for sigma in sigmas])

    images =  []
    with torch.no_grad():
        for s in range(n_steps_each):
            print(s)
            labels = torch.ones(x_mod.shape[0], device=device) * c[1]
            labels = labels.long()
            noise = torch.randn_like(x_mod)
            if beta < 0.15:
                grad1 = score1(x_mod, labels)
                grad2 = score1(x_mod, labels)
            if beta > 15:
                grad1 = score2(x_mod, labels)
                grad2 = score2(x_mod, labels)
            if beta <= 15 and beta >= 0.15:
                grad1 = score1(x_mod, labels)
                grad2 = score2(x_mod, labels)
            x_mod= x_mod + a*1/n_steps_each*grad1 + (1-a)*1/n_steps_each*grad2 + math.sqrt(1/5)*noise

        for s in range(n_steps_each):
            print(s)
            labels = torch.ones(x_mod.shape[0], device=device) * c[s]
            labels = labels.long()
            noise = torch.randn_like(x_mod)
            if beta < 0.15:
                grad1 = score1(x_mod, labels)
                grad2 = score1(x_mod, labels)
            if beta > 15:
                grad1 = score2(x_mod, labels)
                grad2 = score2(x_mod, labels)
            if beta <= 15 and beta >= 0.15:
                grad1 = score1(x_mod, labels)
                grad2 = score2(x_mod, labels)
            images.append(torch.clamp(x_mod, 0.0, 1.0).to(device))
            x_mod= x_mod + a*1/n_steps_each*grad1 + (1-a)*1/n_steps_each*grad2 + math.sqrt(1/n_steps_each)*noise
        
    return images
    
    

############# Generating Samples
n_steps = config["data"]["n_steps"]
imgs = [] 
samples = torch.zeros(config["data"]["grid_size"] ** 2,  config["data"]["channels"], config["data"]["image_size"],
                            config["data"]["image_size"], device=device)
all_samples = SSB(samples, score1, score2, beta, sigmas, n_steps)
############## Saving Images
for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
    sample = sample.view(config["data"]["grid_size"]  ** 2, config["data"]["channels"], config["data"]["image_size"],
                            config["data"]["image_size"])

    image_grid = make_grid(sample, nrow=config["data"]["grid_size"])
    if i % 10 == 0:
        image_grid = image_grid.to(device) 
        im = Image.fromarray((image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()))        
        imgs.append(im)

    save_image(image_grid, os.path.join(os.getcwd(),'Samples', 'image_{}.png'.format(i)), nrow=10)
    torch.save(sample, os.path.join(os.getcwd(),'Samples', 'image_raw_{}.pth'.format(i)))

