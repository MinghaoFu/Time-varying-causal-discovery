import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist

from model.golemmodel import GolemModel
from loss import golem_loss, latent_variable_graphical_lasso_loss
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm
from utils import save_epoch_log, make_optimizer, makedir, check_tensor, linear_regression_initialize
from data import climate, synthetic

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
    parser.add_argument('--latent', action='store_true', help='Existing latent variable or not')
    parser.add_argument('--sparse', action='store_true', help='Sparse matrix or not')
    parser.add_argument('--equal_variances', action='store_true', help='Equal variances or not')
    # model
    parser.add_argument('--embedding_dim', type=int, default=5, help='Embedding dim for t')
    parser.add_argument('--spectral_norm', action='store_true', help='Apply spectral norm to model')
    parser.add_argument('--tol', type=float, default=0, help='Tolerance')
    # loss function, i.e. {'L1': 0.01, 'dag': 0.001, 'grad': 0.1, 'flat': 0.1}
    parser.add_argument('--loss', type=dict, default={'L1': 0.005, 'dag': 0.1, 'flat': 0.0}, help='Embedding dim for t')
    # save
    parser.add_argument('--save_dir', type=str, default='./results/{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), help='Saving directory')
    # generate data
    parser.add_argument('--num', type=int, default=1000, help='Number parameter')
    parser.add_argument('--scale', type=float, default=0.5, help='Variance for gaussian distribution')
    parser.add_argument('--pi', type=float, default=10, help='For DGP,sin(i/pi)')
    parser.add_argument('--distance', type=int, default=2, help='Distance of the largest edge')
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
    
    if torch.cuda.is_available():
        device = torch.device('cuda')  # GPU available
    else:
        device = torch.device('cpu')

    makedir(args.save_dir)
    
    if args.synthetic:
        data, m_true = synthetic.generate_data(args)
    else:
        args.data_path = './data/CESM2_pacific_SST.pkl'
        data, m_true = climate.generate_data(args)

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
    optimizer = make_optimizer(model, args) #optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # if args.pretrain:
    #     model.load_state_dict(torch.load(os.path.join(args.save_dir, f'epoch_{args.pre_epoch}', 'checkpoint.pth')))
    
    if args.train:
        if args.latent:
            print('--- Latent variable modeling')
            preds = np.load('./results/good3/epoch_20000/prediction.npy')
            I = np.identity(preds.shape[1])
            for i in range(args.num):
                preds[i] = np.linalg.inv(I - preds[i]) @ np.linalg.inv(I - preds[i]).T
            S = check_tensor(preds)
            criterion = latent_variable_graphical_lasso_loss(args)
            theta = torch.randn(args.num, args.d, args.d, requires_grad=True)
            L = torch.randn(args.num, args.d, args.d, requires_grad=True)
            alpha = 0.1
            tau = 0.1
            optimizer = optim.SGD([theta, L], lr=1e-3)
            for epoch in range(1000):
                optimizer.zero_grad()
                loss = criterion(S, theta, L, alpha, tau)
                loss['total_loss'].backward()
                optimizer.step()
                if epoch % 100 == 0:
                    print(f'--- Epoch {epoch}, Loss: { {l: loss[l].item() for l in loss.keys()} }')
            import pdb; pdb.set_trace()
            
            
        else:
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