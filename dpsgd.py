import argparse
import torch
import numpy as np
import random

from torch.nn import Module, Linear
from opacus.grad_sample import GradSampleModule

from utils import criterion, clip_grad_by_l2norm, load_tensor_by_batch, str2bool

from loader import load_data

import logging

logging.getLogger().setLevel(logging.INFO)

class LR(Module):
    def __init__(self, n_in, n_out, bias=False):
        super(LR, self).__init__()
        self.linear = GradSampleModule(Linear(n_in, n_out, bias=bias))

    def forward(self, x):
        return self.linear(x)

"""
Example:
    python dpsgd.py --data_type=CA_INCOME --epoch=30 --q=0.001 --lr=0.001 --sigma=1 --device=0 --seed=2 
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device", help="which device to run on, choice from ['CPU', '${INDEX_OF_GPU}']", type=str, default='CPU')
    parser.add_argument("--seed", help="random seed", type=int, default=-1)

    # DATA
    parser.add_argument("--data_type", help="which data to use", type=str)
    parser.add_argument("--split_size", type=float, default=0.2)
    parser.add_argument("--row_clip", help="<0 means no clip for row of data features", type=float, default=-1)

    # TRAIN
    parser.add_argument("--epoch", help="", type=int)
    parser.add_argument("--q", help="sampling rate per iteration", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--eval_freq", type=int, default=1)

    # TRAINING NOISE
    parser.add_argument("--c", help="threshold for gradient clipping(not clip if <=0)", type=float, default=-1)
    parser.add_argument("--perturb_grad", help="inject noise into gradient during local training or not", type=str2bool, default=True)
    parser.add_argument("--sigma", help="scale of local gaussian noise", type=float, default=0.01)

    # clip model weights
    parser.add_argument("--model_c", help="threshold for model clipping(not clip if <=0)", type=float, default=-1)


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
    train_tensor, test_tensor, n, d_in, d_out = load_data(
        data_type=args.data_type, split_size=args.split_size, row_clip=args.row_clip)
    batch_size = int(n*args.q)
    print('batch size per iteration', batch_size)
    test_loader = load_tensor_by_batch(test_tensor)
    train_loader = load_tensor_by_batch(train_tensor, batch_size)
    
    # model
    model = LR(d_in, d_out).to(device)
    # optimizer 
    optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    
    for epoch_i in range(1, args.epoch + 1):
        model.train()
        sum_loss = 0.
            
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # forward
            pred = model(x)
            if len(y.size()) == 1:
                y = torch.unsqueeze(y, dim=-1)
            loss = criterion(pred, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # clip
            if args.c > 0:
                for param in model.parameters():
                    # clip each sample
                    for i, grad_sample in enumerate(param.grad_sample):
                        param.grad_sample[i] = clip_grad_by_l2norm(grad_sample, args.c)
                    # aggregate gradients
                    param.grad.data = torch.mean(param.grad_sample, dim=0)

            # inject noise
            if args.perturb_grad:
                for param in model.parameters():
                    param.grad += 1. / float(len(y)) * torch.normal(mean=0, std=args.sigma, size=param.size(),device=param.device)

            optimizer.step()

            # clear gradient samples to avoid memory leakage
            for param in model.parameters():
                param.grad_sample = None

            sum_loss += loss.item() * len(y)

        # logging.info(f"Epoch #{epoch_i}: Training loss {sum_loss /n}")
            
            
        if epoch_i % args.eval_freq == 0:
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

                logging.info(f"Evaluation at Epoch #{epoch_i}: Test loss {sum_loss / len(test_loader.dataset)}, Test acc {sum_correct / len(test_loader.dataset)}")