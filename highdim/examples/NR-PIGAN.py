import sys

sys.path.append(r'../')
from thop import profile
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import time

from lib.models import NRPIGAN_Generator, NRPIGAN_Discriminator
from lib.data_loader import trainingset_construct_SDE
from lib.visualization import *

import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--data_size', type=int, default=5000)
parser.add_argument('--u_sensor', type=int, default=2)
parser.add_argument('--k_sensor', type=int, default=13)
parser.add_argument('--f_sensor', type=int, default=21)
parser.add_argument('--latent_dim', type=int, default=10)
parser.add_argument('--batch_val', type=int, default=1000)
parser.add_argument('--epoch', type=int, default=10000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--mesh_size', type=int, default=100)
args = parser.parse_args()

if torch.cuda.is_available():
    device_name = 'cuda'
else:
    device_name = 'cpu'
device = torch.device(device_name)
seed = 3407
setup_seed(seed)


def std_cal(A):
    mean = torch.mean(A, dim=0)
    A = (A - mean)
    std = 0
    for i in range(A.size(0)):
        std += torch.norm(A[i, :]) ** 2
    std = torch.sqrt(std / A.size(0))
    return std


def gaussian_noise_measure(input,noise_scale,noise_scale_high,with_noise=False):

    if noise_scale_high is None:
        _noise_scale = noise_scale
    else:
        _noise_scale = np.random.uniform(noise_scale, noise_scale_high, size=input.shape)
    noise = np.random.randn(*input.shape) * _noise_scale
    output = input + noise

    if with_noise:
        return output, noise
    else:
        return output


def uniform_noise_measure(input,noise_scale,with_noise=False):

    noise = ((np.random.rand(*input.shape)* 2.) - 1.) * noise_scale
    output = input + noise
    if with_noise:
        return output, noise
    else:
        return output

def mixture_noise_measure(input, noise_scale_list, mixture_rate_list, with_noise=False):

    data_size, num = input.shape
    rand = np.random.rand(data_size, num)
    cumsum = np.cumsum([0] + mixture_rate_list)
    noise = np.zeros((data_size, num))
    for j, noise_scale in enumerate(noise_scale_list):
        inds = (rand >= cumsum[j]) * (rand < cumsum[j + 1])
        if j == len(noise_scale_list) - 1:
            noise[inds] = ((np.random.rand(np.sum(inds)) * 2) - 1) * noise_scale
        else:
            noise[inds] = np.random.randn(np.sum(inds)) * noise_scale

    output = input + noise

    if with_noise:
        return output, noise
    else:
        return output

def cal_gradient_penalty(dis, G, G_):
    alpha = torch.rand(G.size(0), 1).to(device)
    alpha = alpha.expand(G.size())
    interpolates = (alpha * G + ((1 - alpha) * G_)).requires_grad_(True)
    dis_interpolates = dis(interpolates)
    gradients = torch.autograd.grad(outputs=dis_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(dis_interpolates).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_norm = torch.norm(torch.flatten(gradients, start_dim=1), dim=1)
    loss_gp = torch.mean((grad_norm - 1) ** 2) * 0.1
    return loss_gp

def train(epoch, train_loader, gen, dis, optimizer_gen, optimizer_dis):
    gen.train()
    dis.train()
    loss_d, loss_g = 0, 0
    for _ in range(5):
        for u, k, f, u_coor, k_coor, f_coor in train_loader:
            u = u.to(device)
            k = k.to(device)
            f = f.to(device)
            u_coor = u_coor.to(device)
            k_coor = k_coor.to(device)
            f_coor = f_coor.to(device)
            G = torch.cat((u, k, f), 1)

            optimizer_dis.zero_grad()
            scores_real_D = dis(G)
            z = torch.randn(args.batch_val, args.latent_dim).to(device)
            u_recon, k_recon, f_recon = gen(z, u_coor, k_coor, f_coor)
            G_ = torch.cat((u_recon, k_recon, f_recon), 1)
            scores_fake_D = dis(G_.detach())
            gradient_penalty = cal_gradient_penalty(dis, G, G_)
            loss_dis = -torch.mean(scores_real_D) + torch.mean(scores_fake_D) + gradient_penalty
            loss_dis.backward()
            optimizer_dis.step()
            loss_d = loss_d + loss_dis.item()
            loss_d = loss_d / 5
            if _ == 4:
                optimizer_gen.zero_grad()
                z = torch.randn(args.batch_val, args.latent_dim).to(device)
                u_recon, k_recon, f_recon = gen(z, u_coor, k_coor, f_coor)
                G_ = torch.cat((u_recon, k_recon, f_recon), 1)
                scores = dis(G_)
                loss_gen = -torch.mean(scores)
                loss_gen.backward()
                optimizer_gen.step()
                loss_g = loss_gen.item()
    return loss_d, loss_g

u_data = np.load(file=r'../database/SDE/u_0.08_5000.npy')[0:args.data_size]
k_data = np.load(file=r'../database/SDE/k_0.08_5000.npy')[0:args.data_size]
f_data = np.load(file=r'../database/SDE/f_0.08_5000.npy')[0:args.data_size]

n_validate = 101
test_coor = np.floor(np.linspace(0, 1, n_validate) * args.mesh_size).astype(int)
u_test = u_data[:, test_coor]
k_test = k_data[:, test_coor]
f_test = f_data[:, test_coor]
true_mean_u = torch.mean(torch.from_numpy(u_test), dim=0).type(torch.float).to(device)
true_mean_k = torch.mean(torch.from_numpy(k_test), dim=0).type(torch.float).to(device)
true_std_u = std_cal(torch.from_numpy(u_test)).type(torch.float).to(device)
true_std_k = std_cal(torch.from_numpy(k_test)).type(torch.float).to(device)

nblock = 3
width = 128
nblock_n=3
width_n=128
gen = NRPIGAN_Generator(args.latent_dim, args.u_sensor, args.k_sensor, args.f_sensor, width, nblock,width_n,nblock_n, device)
dis = NRPIGAN_Discriminator(args.u_sensor+args.k_sensor+args.f_sensor, width, device)
optimizer_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.9))
optimizer_dis = optim.Adam(dis.parameters(), lr=args.lr, betas=(0.5, 0.9))

u_coor = np.linspace(-1, 1, args.u_sensor) * np.ones([len(u_data), args.u_sensor])
k_coor = np.linspace(-1, 1, args.k_sensor) * np.ones([len(k_data), args.k_sensor])
f_coor = np.linspace(-1, 1, args.f_sensor) * np.ones([len(f_data), args.f_sensor])

x_u_coor = np.floor(np.linspace(0, 1, args.u_sensor) * args.mesh_size).astype(int)
x_k_coor = np.floor(np.linspace(0, 1, args.k_sensor) * args.mesh_size).astype(int)
x_f_coor = np.floor(np.linspace(0, 1, args.f_sensor) * args.mesh_size).astype(int)

k_training_data = k_data[0:args.data_size, x_k_coor]
u_training_data = u_data[0:args.data_size, x_u_coor]
f_training_data = f_data[0:args.data_size, x_f_coor]

k_training_data = gaussian_noise_measure(k_training_data,noise_scale=0.1,noise_scale_high=None,with_noise=False)

train_loader = trainingset_construct_SDE(u_data=u_training_data, k_data=k_training_data, f_data=f_training_data,
                                             x_u=u_coor, x_k=k_coor, x_f=f_coor, batch_val=args.batch_val)

def main2(seed):
    u_mean_error = []
    u_std_error = []
    k_mean_error = []
    k_std_error = []
    time_history = []
    for epoch in range(1, args.epoch+1):

        if epoch % 100 == 0:
            print('epoch:', epoch)

            with torch.no_grad():
                z = torch.randn(1000, args.latent_dim).to(device)
                coordinate = (torch.linspace(-1, 1, steps=n_validate) * torch.ones((1000, n_validate))).to(device)
                u_recon = gen.reconstruct_u(z, coordinate)
                k_recon = gen.reconstruct_k(z, coordinate)
                mean_u = torch.mean(u_recon, dim=0)
                std_u = std_cal(u_recon)
                mean_k = torch.mean(k_recon, dim=0)
                std_k = std_cal(k_recon)
                mean_L2_error_u = (torch.norm(mean_u - true_mean_u) / torch.norm(true_mean_u)).cpu().numpy()
                std_L2_error_u = (torch.norm(std_u - true_std_u) / torch.norm(true_std_u)).cpu().numpy()
                print('u mean error:', mean_L2_error_u, 'u std error:', std_L2_error_u)

                mean_L2_error_k = (torch.norm(mean_k - true_mean_k) / torch.norm(true_mean_k)).cpu().numpy()
                std_L2_error_k = (torch.norm(std_k - true_std_k) / torch.norm(true_std_k)).cpu().numpy()
                print('k mean error:', mean_L2_error_k, 'k std error:', std_L2_error_k)
                if epoch %100== 0 or epoch == 0:
                    u_mean_error.append(mean_L2_error_u)
                    u_std_error.append(std_L2_error_u)
                    k_mean_error.append(mean_L2_error_k)
                    k_std_error.append(std_L2_error_k)
        time_start = time.time()
        loss_d, loss_g = train(epoch, train_loader, gen, dis, optimizer_gen, optimizer_dis)
        time_stop = time.time()
        time_history.append(time_stop - time_start)
    #np.save(f"results/a=0.08/0.8/u_mean_error.npy", u_mean_error)
    #np.save(f"results/a=0.08/0.8/k_mean_error.npy", k_mean_error)
    #np.save(f"results/a=0.08/0.8/u_std_error.npy", u_std_error)
    #np.save(f"results/a=0.08/0.8/k_std_error.npy", k_std_error)
    #np.save("results/a=0.08/0.8/NRPI_WGAN_time_history", time_history)
    #torch.save(gen, f"results/a=0.08/0.8/gan_{seed}.pth")

    return u_mean_error, u_std_error, k_mean_error, k_std_error

main2(1000)
