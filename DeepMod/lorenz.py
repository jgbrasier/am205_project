import numpy as np
from numpy.lib.function_base import diff
import torch
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp

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
from library import LibraryLorenz

import torch
from torch.utils.data import DataLoader
from itertools import combinations
from functools import reduce

from scipy.integrate import odeint

# Settings for reproducibility
np.random.seed(40)
torch.manual_seed(0)

# lorenz model
def lorenz(t, x, sigma=10, beta=2.66667, rho=28):
	"""
	parameters:

	:sigma : 10
	:beta : 2.7
	:rho : 28
	"""
	return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]

t = np.linspace(0, 5, 1000)
t_range = (t[0], t[-1])
U0= np.array([-8, 8, 27]) # initial point x0,y0
U = solve_ivp(lorenz, t_range, U0, t_eval=t, rtol = 1e-12, method = 'LSODA', atol = 1e-12).y.T


t_norm = t
U_norm = U/np.max(np.abs(U))
# noisify data
noise_lvls = [0.001] #  ,0.5, 0.75, 1.0, 2.0]
all_U_noise = []

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(U_norm[:, 0], U_norm[:, 1],U_norm[:, 2], 'r', label='Input', zorder=0)
plt.savefig('DeepMod/figs/lorenz/lorenz.png')

errs = []

for noise_lvl in noise_lvls:
    noise = noise_lvl*np.random.randn(U.shape[0], U.shape[1])
    U_noise = U + noise
    U_noise_norm = U_noise/np.max(np.abs(U_noise), axis=0)
    all_U_noise.append(U_noise_norm)

    # tensors
    number_of_samples = 500

    idx = np.random.permutation(U_noise.shape[0])
    X = torch.tensor(t_norm.reshape(-1, 1)[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
    y = torch.tensor(U_noise_norm[idx, :][:number_of_samples], dtype=torch.float32)
    print(X.shape, y.shape)

    # Configuration of the function approximator: Here the first argument is the number of input and the last argument the number of output layers.
    network = NN(1, [20, 20, 20, 20, 20], 3)
    # library = Library1D(poly_order=1, diff_order=0)
    library = LibraryLorenz()

    # Configuration of the sparsity estimator and sparsity scheduler used.
    estimator = Threshold(0.5) 
    sparsity_scheduler = TrainTestPeriodic(periodicity=100, patience=500) 
    constraint = LeastSquares() 

    model = DeepMoD(network, library, estimator, constraint)

    # Defining optimizer
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-4)

    # logdir
    logpath = r'DeepMod/logs/lorenz_' + str(noise_lvl)

    train(model, X, y, optimizer, sparsity_scheduler, log_dir=logpath, split=0.8, max_iterations=100000, patience=100, delta=1e-5) 

    print(model.sparsity_masks)
    print(model.estimator_coeffs())

    # config = {'n_in': 1, 'hidden_dims': [20, 20, 20, 20, 20, 20], 'n_out': 2, 'library_function': library_1D_in, 'library_args':{'poly_order': 1, 'diff_order': 0}}
    y_pred = model(X)[0].detach().numpy()


    plt.figure(figsize=(10, 7))
    plt.subplot(311)
    plt.title("Noise level={}".format(noise_lvl))
    plt.plot(t_norm, U_noise_norm[:, 0], 'r', label='Input', zorder=0)
    plt.plot(X.detach().numpy().squeeze(), y_pred[:, 0], 'b.', label='Reconstructed', zorder=1)
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.subplot(312)
    plt.plot(t_norm, U_noise_norm[:, 1]  , 'r', label='Input', zorder=0)
    plt.plot(X.detach().numpy().squeeze(), y_pred[:, 1], 'b.', label='Reconstructed', zorder=1)
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('y')
    plt.subplot(313)
    plt.plot(t_norm, U_noise_norm[:, 2]  , 'r', label='Input', zorder=0)
    plt.plot(X.detach().numpy().squeeze(), y_pred[:, 2], 'b.', label='Reconstructed', zorder=1)
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('y')
    plt.savefig("DeepMod/figs/lorenz/lorenz_pred_{}.png".format(noise_lvl))

    # errors:
    err = np.average(np.sqrt(np.sum((y_pred - y.numpy())**2)/len(y)))
    errs.append(err)


fig, ax = plt.subplots(figsize=(12, 6), ncols=3, nrows=2, subplot_kw={'projection': "3d"})
ax = ax.flatten()
ax[0].plot(U_norm[:, 0], U_norm[:, 1], U_norm[:, 2],  color='blue')
ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$y$')
ax[0].set_title("Input data, no noise")

for i in range(len(noise_lvls)):
    
    ax[i+1].set_title("Input data, noise level={}".format(noise_lvls[i]))
    ax[i+1].plot(all_U_noise[i][:, 0], all_U_noise[i][:, 1], all_U_noise[i][:, 2] , 'b.')
    ax[i+1].set_xlabel('$x$')
    ax[i+1].set_ylabel('$y$')

plt.tight_layout()
plt.savefig("DeepMod/figs/lorenz/lorenz_sim.png")

plt.figure()
plt.title("RMSE vs noise")
plt.plot(noise_lvls, errs, '-ok')
plt.xlabel("noise level")
plt.ylabel("average RMSE")
plt.savefig("DeepMod/figs/lorenz/lorenz_err.png")