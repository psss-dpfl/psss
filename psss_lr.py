import argparse
import torch
import numpy as np
import random

from torch.nn import Module, Linear

from utils import criterion, clip_grad_by_l2norm, load_tensor_by_batch 

from loader import load_data

import logging
import time
import math 

GAUSSIAN_APPRO=True

logging.getLogger().setLevel(logging.INFO)

class LR(Module):
    def __init__(self, n_in, n_out, bias=False):
        super(LR, self).__init__()
        self.linear = Linear(n_in, n_out, bias=bias)

    def forward(self, x):
        return self.linear(x)
    
def discretize_tensor(x, gamma):
    scaled_x = gamma * x
    floor = torch.floor(scaled_x)
    p_ceil = scaled_x - floor
    
    random_nums = torch.rand(size=scaled_x.size(), device=scaled_x.device, requires_grad=False)
    choice_floor = (random_nums > p_ceil).type(torch.float32)

    discrete_x = choice_floor * floor + (1. - choice_floor) * (floor + 1.)

    return discrete_x

def skellam_noise(size, poisson_mu, device):
    rate = torch.full(size, poisson_mu, dtype=torch.float).to(device)
    noise = torch.poisson(rate) - torch.poisson(rate)

    return noise#.to(device)

def noisy_gradient_exact(gamma, weight, x, targets, mu_multiplier, device):
    dis_weight = discretize_tensor(weight, gamma).reshape(-1,1) 
    dis_x = discretize_tensor(x, gamma)  
    dis_targets = discretize_tensor(targets, gamma) 

    f1 =  dis_x / gamma
    mu1 = mu_multiplier*(gamma**2)
    sum = 1/2 * (f1 + skellam_noise(f1.size(), mu1, device)) 
    f2 = torch.matmul(dis_x,dis_weight)*dis_x
    mu2 = mu_multiplier*(gamma**6)
    sum = sum + 1/4 * (f2 + skellam_noise(f2.size(), mu2, device))
    f3 = -dis_targets * dis_x 
    mu3 = mu_multiplier*(gamma**4)
    sum = sum + f3 + skellam_noise(f3.size(), mu3, device)
       
    return sum

def noisy_gradient_approx(gamma, weight, x, targets, mu_multiplier, device):
    dis_weight = discretize_tensor(weight, gamma).reshape(-1,1) / gamma
    dis_x = discretize_tensor(x, gamma) / gamma 
    dis_targets = discretize_tensor(targets, gamma) / gamma
     
    f1 =  dis_x 
    sum = 1/2  * (f1 + torch.empty_like(f1,device=device).normal_(std=math.sqrt(2*mu_multiplier)))
    f2 = torch.matmul(dis_x,dis_weight)*dis_x
    sum = sum + 1/4  * (f2 + torch.empty_like(f2,device=device).normal_(std=math.sqrt(2*mu_multiplier)))
    f3 = -dis_targets * dis_x
    sum = sum + f3 + torch.empty_like(f3,device=device).normal_(std=math.sqrt(2*mu_multiplier))
        
    return sum

"""
Example:
    python psss_lr.py --data_type=TX_INCOME --epoch=20 --q=0.001 --lr=0.01 --mu_multiplier=0.2639 --seed=2 --device=0 
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device", help="which device to run on, choice from ['CPU', '${INDEX_OF_GPU}']", type=str, default="0")
    parser.add_argument("--seed", help="random seed", type=int, default=2)

    # DATA
    parser.add_argument("--data_type", help="which data to use", type=str, default="TX_INCOME")
    parser.add_argument("--split_size", type=float, default=0.2)
    parser.add_argument("--row_clip", help="<0 means no clip for row of data features", type=float, default=1)

    # TRAIN
    parser.add_argument("--epoch", help="", type=int, default=30)
    parser.add_argument("--q", help="sampling rate per iteration", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--eval_freq", type=int, default=1)
    
    # TRAINING NOISE
    parser.add_argument("--mu_multiplier", help="scale of mu_multiplier", type=float, default=10)

    # clip model weights
    parser.add_argument("--model_c", help="threshold for model clipping(not clip if <=0)", type=float, default=1)

    args = parser.parse_args()

    print(args)
    
    # fix seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # get device
    if args.device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    
    # load data
    logging.info(f"loading data...")

    train_tensor, test_tensor, n, d_in, d_out = load_data(
        data_type=args.data_type, split_size=args.split_size, row_clip=args.row_clip)
    batch_size = int(n*args.q)
    logging.info(f"loading tensor by batch...")
    print('batch size per iteration', batch_size)
    test_loader = load_tensor_by_batch(test_tensor)
    train_loader = load_tensor_by_batch(train_tensor, batch_size)
    logging.info(f"training start...")

    for gamma in [262144, 65536, 16384, 4096, 1024]:
    
        # model
        model = LR(d_in, d_out).to(device)
        # optimizer 
        optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
               
        for epoch_i in range(1, args.epoch + 1):
            model.train()
            sum_loss = 0.
            start_time = time.time()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                # forward
                pred = model(x)
                if len(y.size()) == 1:
                    y = torch.unsqueeze(y, dim=-1)
                loss = criterion(pred, y)

                loss.backward()
                for param in model.parameters():
                    if GAUSSIAN_APPRO:
                        noisy_grad = noisy_gradient_approx(gamma, param, x, y, args.mu_multiplier, device)
                        noisy_grad = noisy_grad.sum(dim=0).reshape(1,-1) 
                        param.grad = noisy_grad 
                    else:
                        noisy_grad = noisy_gradient_exact(gamma, param, x, y, args.mu_multiplier, device)
                        noisy_grad = noisy_grad.sum(dim=0).reshape(1,-1) 
                        param.grad = noisy_grad            

                optimizer.step()

                # clip model weights
                if args.model_c > 0:
                    for param in model.parameters():
                        param = clip_grad_by_l2norm(param, args.model_c)
                
                # clear gradient samples to avoid memory leakage
                for param in model.parameters():
                    param.grad_sample = None

                sum_loss += loss.item() * len(y)
                
            with torch.no_grad():
                model.eval()
                sum_loss, sum_correct = 0., 0.
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)

                    pred = model(x)
                    if len(y.size()) == 1:
                        y = torch.unsqueeze(y, dim=-1)
                    loss = criterion(pred, y)

                    predict = 1. / (1. + torch.exp(- pred))

                    sum_correct += torch.sum(torch.sign(predict - 0.5) * (y - 0.5) * 2 == 1).item()
                    sum_loss += loss.item() * len(y)

                logging.info(f"Gamma is #{gamma}, mu is #{args.mu_multiplier}, Task is #{args.data_type}: Epoch #{epoch_i}: Test loss {sum_loss / len(test_loader.dataset)}, Test acc {sum_correct / len(test_loader.dataset)}")

            # logging.info('one epoch costs %.5f', time.time()-start_time)