__all__  = ['NRPIGAN_Generator', 'NRPIGAN_Discriminator']
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class plainBlock(nn.Module):

    def __init__(self, width):
        super(plainBlock, self).__init__()
        self.fc1 = nn.Linear(width,width)
    def forward(self, x):
        out = self.fc1(x)
        out = torch.tanh(out)
        return out

def gaussian_reparameterize(input, eps=None):
    if eps is None:
        eps = torch.randn_like(input)
    return eps * input

def uniform_reparameterize(input,eps=None):
    if eps is None:
        eps = torch.rand_like(input) * 2 - 1
    return  eps * input

def mixture_reparameterize(input):
    mixture_rate_list = [0.7, 0.2, 0.1]
    device = input.device

    rand = torch.rand_like(input)
    cumsum = torch.cumsum(torch.tensor([0.0] + mixture_rate_list), dim=0)
    eps = torch.zeros_like(input)
    for j in range(len(mixture_rate_list)):
        inds = (rand >= cumsum[j]) * (rand < cumsum[j + 1])
        if j == len(mixture_rate_list) - 1:
            eps[inds] = ((torch.rand((torch.sum(inds)),device=device) * 2) - 1)
        else:
            eps[inds] = torch.randn(torch.sum(inds),device=device)

    return eps * input


class NRPIGAN_Generator(nn.Module):
    def __init__(self, lat_dim, udata_dim, kdata_dim, fdata_dim, width, n_blocks,width_n,n_blocks_n, device):
        super(NRPIGAN_Generator, self).__init__()
        self.device = device

        self.u_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.u_gen = nn.Sequential(nn.Linear(lat_dim + 1, width), self.u_blocks, nn.Linear(width, 1)).to(device)

        self.k_blocks = nn.Sequential(*(n_blocks * [plainBlock(width)]))
        self.k_gen = nn.Sequential(nn.Linear(lat_dim + 1, width), self.k_blocks, nn.Linear(width, 1)).to(device)

        self.nk_blocks = nn.Sequential(*(n_blocks_n * [plainBlock(width_n)]))
        self.nk_gen = nn.Sequential(nn.Linear(lat_dim + 1, width_n), self.nk_blocks, nn.Linear(width_n, 1)).to(device)

        self.udata_dim = udata_dim
        self.kdata_dim = kdata_dim
        self.fdata_dim = fdata_dim
        self.lat_dim = lat_dim

    def combine_xz(self, x, z):
        x_new = x.view(-1, 1).to(self.device)
        z_new = torch.repeat_interleave(z, x.size(1), dim=0).to(self.device)  # .view(-1,self.latent_dim)
        return torch.cat((x_new, z_new), 1)

    def reconstruct_u(self, z, ucoor):
        x_u = self.combine_xz(ucoor, z)
        urecon = self.u_gen(x_u).view(-1, ucoor.size(1))
        return urecon

    def reconstruct_k(self,z,kcoor):
        x_k=self.combine_xz(kcoor,z)
        krecon=self.k_gen(x_k).view(-1, kcoor.size(1))
        return krecon

    def construct_nk(self,z,kcoor):
        x_k = self.combine_xz(kcoor, z)
        n = self.nk_gen(x_k).view(-1, kcoor.size(1))
        n = gaussian_reparameterize(n)
        return n

    def f_recontruct(self, z, fcoor):
        x = Variable(fcoor.view(-1, 1).type(torch.FloatTensor), requires_grad=True).to(self.device)
        z_uk = torch.repeat_interleave(z, fcoor.size(1), dim=0)

        x_PDE = torch.cat((x, z_uk), 1)
        u_PDE = self.u_gen(x_PDE)
        k_PDE = self.k_gen(x_PDE)

        # calculate derivative
        u_PDE_x = torch.autograd.grad(outputs=u_PDE, inputs=x, grad_outputs=torch.ones(u_PDE.size()).to(self.device),
                                      create_graph=True, only_inputs=True)[0]
        u_PDE_xx = torch.autograd.grad(outputs=u_PDE_x, inputs=x, grad_outputs=torch.ones(u_PDE_x.size()).to(self.device),
                                       create_graph=True, only_inputs=True)[0]
        k_PDE_x = torch.autograd.grad(outputs=k_PDE, inputs=x, grad_outputs=torch.ones(k_PDE.size()).to(self.device),
                                      create_graph=True, only_inputs=True)[0]
        f_recon = -0.1 * (k_PDE_x * u_PDE_x + k_PDE * u_PDE_xx).view(-1, fcoor.size(1))
        return f_recon

    def forward(self, z, ucoor, kcoor, fcoor):
        urecon = self.reconstruct_u(z, ucoor)
        krecon=self.reconstruct_k(z,kcoor)
        kn=self.construct_nk(z,kcoor)
        krecon=kn+krecon
        f_recon = self.f_recontruct(z, fcoor)
        return urecon, krecon, f_recon

class NRPIGAN_Discriminator(nn.Module):
    def __init__(self, in_dim, width, device):
        super(NRPIGAN_Discriminator, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(in_dim, width).to(device)
        self.fc2 = nn.Linear(width, width).to(device)
        self.fc3 = nn.Linear(width, width).to(device)
        self.fc4 = nn.Linear(width, width).to(device)
        self.fc5 = nn.Linear(width, 1).to(device)

    def forward(self, x):
        x = torch.tanh(self.fc1(x)).to(self.device)
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x
