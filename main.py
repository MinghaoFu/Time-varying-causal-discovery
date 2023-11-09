import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from data import synthetic_data

from model import GolemModel, iVAE, joint_gaussian, latent_joint_gaussian

from loss import golem_loss, latent_variable_graphical_lasso_loss
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm
from utils import save_epoch_log, make_optimizer, makedir, check_tensor, \
    linear_regression_initialize, postprocess, make_dots, is_markov_equivalent
from data import climate, synthetic_data

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
    parser = argparse.ArgumentParser(description="Example script with argparse")
    parser.add_argument('--train', action='store_true', help='Train or not')
    parser.add_argument('--pretrain', action='store_true', help='Pre Train or not')
    parser.add_argument('--ddp', action='store_true', help='Training with distributed data parallel or not')
    parser.add_argument('--pre_epoch', type=int, default=0, help='The checkpoint ID for pretrain')
    parser.add_argument('--epoch', type=int, default=1000, help='Epochs')
    parser.add_argument('--init_epoch', type=int, default=200, help='Epochs for initialization')
    parser.add_argument('--batch_size', type=int, default=100000, help='Batch size')
    parser.add_argument('--lag', type=int, default=10, help='Lag value')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data or not')
    parser.add_argument('--time_varying', action='store_true', help='Time-varying or static causal discovery.')
    parser.add_argument('--sparse', action='store_true', help='Sparse matrix or not')
    parser.add_argument('--noise_type', type=str, default='gaussian_ev', help='Noise type')
    parser.add_argument('--seed', type=int, default=69, help='Random seed')
    parser.add_argument('--gt_init', action='store_true', help='Initialization with ground truth or not')
    # model
    parser.add_argument('--embedding_dim', type=int, default=5, help='Embedding dim for t')
    parser.add_argument('--spectral_norm', action='store_true', help='Apply spectral norm to model')
    parser.add_argument('--tol', type=float, default=0, help='Tolerance')
    parser.add_argument('--graph_thres', type=float, default=0.3, help='Threshold for generating dag')
    # loss function, i.e. {'L1': 0.01, 'dag': 0.001, 'grad': 0.1, 'flat': 0.1}
    parser.add_argument('--loss', type=dict, default={'L1': 0.005, 'dag': 0.1, 'flat': 0.0}, help='Embedding dim for t')
    parser.add_argument('--sparsity_M', type=float, default=0.8, help='Sparsity of causal effects in measured variables.')
    parser.add_argument('--sparsity_L', type=float, default=0.8, help='Sparsity of causal effects from latent to measured.')
    parser.add_argument('--DAG', type=float, default=0.8, help='Lambda DAG.')
    # save
    parser.add_argument('--save_dir', type=str, default='./results/{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), help='Saving directory')
    # generate data
    parser.add_argument('--num', type=int, default=1000, help='Number parameter')
    parser.add_argument('--scale', type=float, default=0.5, help='Variance for gaussian distribution')
    parser.add_argument('--pi', type=float, default=10, help='For DGP,sin(i/pi)')
    parser.add_argument('--distance', type=int, default=2, help='Distance of the largest edge')
    parser.add_argument('--max_d_L', type=int, default=1, help='Number of maximum latent variable')
    parser.add_argument('--d_L', type=int, default=1, help='Number of latent variable')
    parser.add_argument('--d_X', type=int, default=10, help='Number of measured variable')
    
    parser.add_argument('--degree', type=int, default=2, help='Graph degree')
    parser.add_argument('--condition', type=str, default='necessary', help='Causal graph condition')
    # training lr
    parser.add_argument('--decay_type', type=str, default='step', choices=['step', 'multi step', 'cosine'], help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD', 'RMSprop'], help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gradient_noise', type=float, default=None, help='Variance of gradient noise')
    # step decay, (or cosine annealing decay)
    parser.add_argument('--step_size', type=int, default=1000, help='After step_size epochs, lr = gamma * lr, or cosine annealing start')
    parser.add_argument('--gamma', type=float, default=0.5, help='After step_size epochs, lr = gamma * lr')
    # multi step decay
    parser.add_argument('--decay', type=list, default=[200, 400, 800, 1000], help='Epoch milestones for lr = gamma * lr')
    # Adam, RMSprop
    parser.add_argument('--betas', nargs=2, type=float, default=[0.9, 0.999], help='Betas for optimizer')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon for optimizer')
    # SGD
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    
    args = parser.parse_args()
    
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
            B = check_tensor(dataset.B, dtype=torch.float32)

            import pdb; pdb.set_trace()
            I = check_tensor(torch.eye(args.d_X))
            T = check_tensor(torch.arange(args.num) / args.num, dtype=torch.float32).reshape(-1, 1)
            model = iVAE(args)
            if torch.cuda.is_available():
                model = model.cuda()
            if args.optimizer == 'ADAM':
                optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon, weight_decay=0) # make_optimizer(model, args)
            elif args.optimizer == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=args.lr)

            for epoch in range(args.epoch):
                model.train()
                optimizer.zero_grad()
                loss, z = model.elbo(X, T)
                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(), 5.)
                
                if args.gradient_noise is not None:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad += check_tensor(torch.normal(0, args.gradient_noise, size=param.grad.size()))
                
                # for i in range(model.B.shape[0]):
                #     model.B.grad[i, i] = 0
                
                optimizer.step()
                

                if (epoch % (args.epoch // 100) == 0): # or markov_equivalence(dataset.B.T, est_B_postprocess.T)
                    model.eval()
                    print(loss.item())
                #     print(f'--- Epoch {epoch}, Loss: { {l: loss[l].item() for l in loss.keys()} }') #lr: {optimizer.get_lr()}
                #     print('--- Estimated covariance - sample covariance = {}'.format(torch.norm(model.est_X_cov - X_cov).item()))
                #     est_B = model.B.cpu().detach().numpy()
                #     est_B_postprocess = postprocess(est_B, graph_thres=args.graph_thres)
                #     est_EX_cov = model.EX_cov.cpu().detach().numpy()
                #     fig_dir = os.path.join(args.save_dir, 'figs')
                #     makedir(fig_dir)
                #     B_labels = [f'M{i}' for i in range(dataset.B.shape[1])]
                #     make_dots(dataset.B, B_labels, fig_dir, 'B')
                #     make_dots(est_B, B_labels, fig_dir, 'est_B')
                #     make_dots(est_B_postprocess, B_labels, fig_dir, 'est_B_postprocess')
                #     # print(dataset.B)
                #     # print(est_B_postprocess)
                #     # print(C, est_C)
                #     if args.d_L != 0:
                #         L_labels = [f'L{i}' for i in range(dataset.C.shape[1])]
                #         est_C = model.C.cpu().detach().numpy()
                #         est_C[np.abs(est_C) <= args.graph_thres] = 0
                #         est_EL_cov = model.EL_cov
                #         est_BC = np.concatenate((np.zeros((args.max_d_L, args.max_d_L + args.d_X)), np.concatenate((est_C, est_B_postprocess), axis=1)), axis=0)
                #         make_dots(dataset.BC, L_labels + B_labels, fig_dir, 'BC')
                #         make_dots(est_BC, L_labels + B_labels, fig_dir, 'est_BC')
        else:
            args.data_path = './data/CESM2_pacific_SST.pkl'
            data = climate.generate_data(args)

            T = args.num * np.arange(data.shape[0]) / data.shape[0]

            T_tensor = check_tensor(T)
            data_tensor = check_tensor(data)
            data_tensor_label = check_tensor(m_true)
            B_init = linear_regression_initialize(data, args.distance)
            #print('B initialization: \n{}'.format(B_init))
            
            dataset = CustomDataset(data_tensor, T_tensor, data_tensor_label)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            model = GolemModel(args, data.shape[1], device, equal_variances=False)
            if args.ddp:
                model = nn.DataParallel(model, range(2))
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) #make_optimizer(model, args)
            
            if args.train:
                criterion = golem_loss(args)
                train_model(args, model, criterion, data_loader, optimizer, m_true, data, T_tensor, B_init, pre_epoch=args.pre_epoch)
                save_epoch_log(args, model, m_true, data, T_tensor, args.epoch)
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{args.epoch}', 'checkpoint.pth'))
                model.load_state_dict(torch.load(os.path.join(args.save_dir, f'epoch_{args.epoch}', 'checkpoint.pth')))

            else:
                model.load_state_dict(torch.load(os.path.join(args.save_dir, f'epoch_{args.pre_epoch}', 'checkpoint.pth')))
                model.eval()
            
if __name__ == "__main__":
    main()