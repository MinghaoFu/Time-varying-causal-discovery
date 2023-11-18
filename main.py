import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ipdb

from torch.utils.data import TensorDataset, DataLoader
from config import *
from data import synthetic_data
from model import GolemModel, iVAE, joint_gaussian, latent_joint_gaussian
from loss import golem_loss, latent_variable_graphical_lasso_loss
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm
from utils import save_epoch_log, make_optimizer, makedir, check_tensor, \
    linear_regression_initialize, postprocess, make_dots, is_markov_equivalent
from data import climate, synthetic_data

from model.dyn_vae import TV_VAE

np.set_printoptions(precision=3, suppress=True)
np.random.seed(100)

class CustomDataset(Dataset):
    def __init__(self, X, T, B):
        self.X = X
        self.T = T
        self.B = B

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.B[idx]

def train_model(args, model, criterion, data_loader, optimizer, m_true, X, T_tensor, B_init, pre_epoch=0):
    model.train()  
    if args.pretrain:
        
        for epoch in tqdm(range(args.init_epoch)):
            for batch_X, T, _ in data_loader:
                optimizer.zero_grad()  
                B = model(T)
                B_label = check_tensor(B_init).repeat(T.shape[0], 1, 1)
                loss = criterion(batch_X, T, B, B_label)
                loss['total_loss'].backward()
                optimizer.step()

        save_epoch_log(args, model, m_true, X, T_tensor, -1)
        print(f"--- Init F based on linear regression, ultimate loss: {loss['total_loss'].item()}")
        
    for epoch in range(args.epoch):
        model.train()
        if epoch < pre_epoch:
            continue
        for batch in data_loader:
            X, T, B_label = batch
            optimizer.zero_grad()  
            B = model(T) 
            loss = criterion(X, T, B)
            loss['total_loss'].backward()
            if args.gradient_noise is not None:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += check_tensor(torch.normal(0, args.gradient_noise, size=param.grad.size()))
            optimizer.step()
        
        if epoch % (args.epoch // 100) == 0:
            save_epoch_log(args, model, m_true, X, T_tensor, epoch)
            print(f'--- Epoch {epoch}, Loss: { {l: loss[l].item() for l in loss.keys()} }')
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{epoch}', 'checkpoint.pth'))
            
        #optimizer.schedule()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='cheetah_run')
    
    args = load_yaml('./config.yaml')

    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')  # GPU available
    else:
        device = torch.device('cpu')

    makedir(args.save_dir)
    
    if args.time_varying is False:  
        dataset = synthetic_data.generate_data(args)
        X = check_tensor(dataset.X, dtype=torch.float32)
        X = X - X.mean(dim=0)
        B = check_tensor(dataset.B, dtype=torch.float32)
        I = check_tensor(torch.eye(args.d_X))
        inv_I_minus_B = torch.inverse(check_tensor(I - B))
        X_cov = torch.cov(X.T)
        if args.d_L > 0:
            C = check_tensor(dataset.C, dtype=torch.float32)
            BC = check_tensor(dataset.BC)#np.concatenate((np.zeros((args.max_d_L, args.max_d_L + args.d_X)), np.concatenate((C, B), axis=1)), axis=0)
            model = latent_joint_gaussian(args, dataset)
            est_X_cov, nll = model.log_gaussian_likelihood(B, C, check_tensor(dataset.EX_cov, dtype=torch.float32), check_tensor(dataset.EL_cov, dtype=torch.float32), args.num, X_cov)
        elif args.d_L == 0:
            model = joint_gaussian(args)
            est_X_cov, nll = model.log_gaussian_likelihood(B, check_tensor(dataset.EX_cov, dtype=torch.float32), args.num, X_cov)
    
        print('--- Population covariance - sample covariance = {}'.format(torch.norm(est_X_cov - X_cov).item()))
        print('--- True nll: {}'.format(nll.item()))

        if torch.cuda.is_available():
            model = model.cuda()
        if args.optimizer == 'ADAM':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon, weight_decay=0) # make_optimizer(model, args)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)

        for epoch in range(args.epoch):
            model.train()
            optimizer.zero_grad()
            loss = model(args, X_cov)
            loss['score'].backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 5.)
            
            if args.gradient_noise is not None:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += check_tensor(torch.normal(0, args.gradient_noise, size=param.grad.size()))
            
            for i in range(model.B.shape[0]):
                model.B.grad[i, i] = 0
            
            optimizer.step()

            if (epoch % (args.epoch // 100) == 0): # or markov_equivalence(dataset.B.T, est_B_postprocess.T)
                model.eval()
                print(f'--- Epoch {epoch}, Loss: { {l: loss[l].item() for l in loss.keys()} }') #lr: {optimizer.get_lr()}
                print('--- Estimated covariance - sample covariance = {}'.format(torch.norm(model.est_X_cov - X_cov).item()))
                est_B = model.B.cpu().detach().numpy()
                est_B_postprocess = postprocess(est_B, graph_thres=args.graph_thres)
                est_EX_cov = model.EX_cov.cpu().detach().numpy()
                fig_dir = os.path.join(args.save_dir, 'figs')
                makedir(fig_dir)
                B_labels = [f'M{i}' for i in range(dataset.B.shape[1])]
                make_dots(dataset.B, B_labels, fig_dir, 'B')
                make_dots(est_B, B_labels, fig_dir, 'est_B')
                make_dots(est_B_postprocess, B_labels, fig_dir, 'est_B_postprocess')
                # print(dataset.B)
                # print(est_B_postprocess)
                # print(C, est_C)
                if args.d_L != 0:
                    L_labels = [f'L{i}' for i in range(dataset.C.shape[1])]
                    est_C = model.C.cpu().detach().numpy()
                    est_C[np.abs(est_C) <= args.graph_thres] = 0
                    est_EL_cov = model.EL_cov
                    est_BC = np.concatenate((np.zeros((args.max_d_L, args.max_d_L + args.d_X)), np.concatenate((est_C, est_B_postprocess), axis=1)), axis=0)
                    make_dots(dataset.BC, L_labels + B_labels, fig_dir, 'BC')
                    make_dots(est_BC, L_labels + B_labels, fig_dir, 'est_BC')
                    
    else:
        if args.synthetic:
            dataset = synthetic_data.generate_data(args)
            X = check_tensor(dataset.X, dtype=torch.float32)
            X = X - X.mean(dim=0)
            T = check_tensor(torch.arange(args.num) / args.num, dtype=torch.float32).reshape(-1, 1)
            model = TV_VAE(args)
            tensor_dataset = TensorDataset(X, T)
            data_loader = DataLoader(tensor_dataset, batch_size=args.batch_size, shuffle=False)
            if torch.cuda.is_available():
                model = model.cuda()
            if args.optimizer == 'ADAM':
                print(type(args.lr))
                optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon, weight_decay=0) # make_optimizer(model, args)
            elif args.optimizer == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=args.lr)
            
            
            for epoch in range(args.epoch):
                for X_batch, T_batch in data_loader:
                    model.train()
                    optimizer.zero_grad()
                    losses = model.compute_vae_loss(X_batch, T_batch)
                    losses['total_loss'].backward()
                    
                    if args.gradient_noise is not None:
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad += check_tensor(torch.normal(0, args.gradient_noise, size=param.grad.size()))
            
                    optimizer.step()

                if (epoch % (args.epoch // 100) == 0): # or markov_equivalence(dataset.B.T, est_B_postprocess.T)
                    print(f'--- Epoch {epoch}, Loss: { {l: losses[l].item() for l in losses.keys()} }')
                    model.eval()
                    BT, CT, CT_1 = model(X, T)
                    BT, CT, CT_1 = BT.cpu().detach().numpy(), CT.cpu().detach().numpy(), CT_1.cpu().detach().numpy()
                    epoch_save_dir = os.path.join(args.save_dir, f'Epoch {epoch}')
                    makedir(epoch_save_dir)
                    
                    for i in range(5):
                        sub_fig_dir = os.path.join(epoch_save_dir, f't_{i}')
                        Bt = postprocess(BT[i], graph_thres=args.graph_thres)
                        Ct = CT[i]
                        Ct[Ct < args.graph_thres] = 0
                        B_labels = [f'M{i}' for i in range(args.d_X)]
                        C_labels = [f'L{i}' for i in range(args.d_L)]
                        makedir(sub_fig_dir)
                        # make_dots(dataset.B[i], B_labels, sub_fig_dir, f'B_gt{i}')
                        # make_dots(dataset.C[i], C_labels, sub_fig_dir, f'C_gt{i}')
                        BCt = np.concatenate((np.zeros((args.d_L, args.d_L + args.d_X)), np.concatenate((Ct, Bt), axis=1)), axis=0)
                        make_dots(dataset.BCs[i], C_labels + B_labels, sub_fig_dir, f'BC_gt{i}')
                        make_dots(BCt, C_labels + B_labels, sub_fig_dir, f'BC_est{i}')
                        if args.assume_time_lag:
                            Ct_1 = CT_1[i][CT_1[i] < args.graph_thres]

        else:
            args.data_path = './data/CESM2_pacific_SST.pkl'
            data = climate.generate_data(args)

            T = args.num * np.arange(data.shape[0]) / data.shape[0]

            T_tensor = check_tensor(T)
            data_tensor = check_tensor(data)
            B_init = linear_regression_initialize(data, args.distance)
            #print('B initialization: \n{}'.format(B_init))
            
if __name__ == "__main__":
    # import traceback
    # try:
    main()
    # except Exception as e:
    #     traceback.print_exc()
    #     print(f"Exception occurred: {e}")
    #     ipdb.post_mortem()
        
