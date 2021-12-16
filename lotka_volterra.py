import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# DeepMoD functions
from deepymod import DeepMoD
from deepymod.model.func_approx import NN
from deepymod.model.constraint import LeastSquares
from deepymod.model.sparse_estimators import Threshold, PDEFIND
from deepymod.training.sparsity_scheduler import TrainTestPeriodic
from deepymod.model.library import Library1D
from scipy.io import loadmat

# custom train loop
from train import train
# custom library
from library import LibraryLotkaVoltera

import torch
from torch.utils.data import DataLoader
from itertools import combinations
from functools import reduce

from scipy.integrate import odeint

# Settings for reproducibility
np.random.seed(40)
torch.manual_seed(0)

# lotka voltera model
a = 1.
b = 0.5
c = 0.5
d = 2.
def lotkavolterra(U, t=0):
    """ Return the growth rate of predator and prey populations. 
    U: [x, y] population vector
    """
    return np.array([ a*U[0] -   b*U[0]*U[1] ,
                  -c*U[1] + d*b*U[0]*U[1] ])
t = np.linspace(0, 15,  500)              # time
U0 = np.array([10, 5])                     # initials conditions: 10 prey and 5 predator
U = odeint(lotkavolterra, U0, t)

# def dU_dt(U, t):
#     return [U[1], -1*U[1] - 5*np.sin(U[0])]
# U0 = [2.5, 0.4]

# t = np.linspace(0, 5,  100)              # time
# U = odeint(dU_dt, U0, t)

t_norm = t
U_norm = U
# noisify data
noise_lvls = [0.1, 1.0, 2.5, 5.0]
for noise_lvl in noise_lvls:
    noise = 0.1*np.random.randn(U.shape[0], U.shape[1])
    U_noise = U + noise
    U_noise_norm = U_noise

    normalise = True
    if normalise:
        # t_norm = t/np.max(np.abs(t))
        U_norm = U/np.max(np.abs(U))
        U_noise_norm = U_noise/np.max(np.abs(U_noise))

    plt.figure()
    plt.subplot(211)
    plt.plot(U_norm[:, 0], U_norm[:, 1], color='blue')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Input data, no noise")
    plt.subplot(212)
    plt.title("DeepMod reconstructed system, noise level={}".format(noise_lvl))
    plt.plot(U_noise_norm[:, 0], U_noise_norm[:, 1]  , color='blue')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('y')
    plt.ylabel('population')
    plt.tight_layout()
    plt.savefig("./figs/lv/lotkavolterra_sim_{}.png".format(noise_lvl))


    # tensors
    number_of_samples = 250

    idx = np.random.permutation(U_noise.shape[0])
    X = torch.tensor(t_norm.reshape(-1, 1)[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
    y = torch.tensor(U_noise_norm[idx, :][:number_of_samples], dtype=torch.float32)
    print(X.shape, y.shape)

    train_dataloader = DataLoader(X, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(y, batch_size=64, shuffle=True)

    # Configuration of the function approximator: Here the first argument is the number of input and the last argument the number of output layers.
    network = NN(1, [20, 20, 20, 20, 20, 20], 2)
    # library = Library1D(poly_order=1, diff_order=0)
    library = LibraryLotkaVoltera()

    # Configuration of the sparsity estimator and sparsity scheduler used.
    estimator = Threshold(0.5) 
    sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=500) 
    constraint = LeastSquares() 

    model = DeepMoD(network, library, estimator, constraint)

    # Defining optimizer
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-4)

    # logdir
    logpath = r'./logs/lotkavoltera_' + str(noise_lvl)

    train(model, X, y, optimizer, sparsity_scheduler, log_dir=logpath, split=0.8, max_iterations=100000, patience=200, delta=1e-5) 

    print(model.sparsity_masks)
    print(model.estimator_coeffs())

    # config = {'n_in': 1, 'hidden_dims': [20, 20, 20, 20, 20, 20], 'n_out': 2, 'library_function': library_1D_in, 'library_args':{'poly_order': 1, 'diff_order': 0}}
    y_pred = model(X)[0].detach().numpy()


    plt.figure()
    plt.subplot(221)
    plt.title("Noise level=".format(noise_lvl))
    plt.plot(t_norm, U_noise_norm[:, 0], 'r', label='Input')
    plt.plot(X.detach().numpy().squeeze(), y_pred[:, 0], 'b--', label='Reconstructed', zorder=1)
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.subplot(222)
    plt.plot(t_norm, U_noise_norm[:, 1]  , 'r', label='Input')
    plt.plot(X.detach().numpy().squeeze(), y_pred[:, 1], 'b--', label='Reconstructed', zorder=1)
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('y')
    plt.savefig("./figs/lv/lotkavolterra_pred_{}.png".format(noise_lvl))