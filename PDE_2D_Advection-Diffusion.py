# General imports
import numpy as np
import torch
import matplotlib.pylab as plt

# DeepMoD functions

from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.model.func_approx import NN
from deepymod.model.library import Library2D
from deepymod.model.constraint import LeastSquares
from deepymod.model.sparse_estimators import Threshold, PDEFIND
from deepymod.training import train
from deepymod.training.sparsity_scheduler import TrainTestPeriodic
from scipy.io import loadmat

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)

# Configuring GPU or CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)


def create_data():
    data = loadmat("data/advection_diffusion.mat")
    usol = np.real(data["Expression1"]).astype("float32")
    usol = usol.reshape((51, 51, 61, 4))
    x_v = usol[:,:,:,0]
    y_v = usol[:,:,:,1]
    t_v = usol[:,:,:,2]
    u_v = usol[:,:,:,3]
    coords = torch.from_numpy(np.stack((t_v,x_v, y_v), axis=-1))
    data = torch.from_numpy(usol[:, :, :, 3]).unsqueeze(-1)
    # alternative way of providing the coordinates
    # coords = torch.from_numpy(np.transpose((t_v.flatten(), x_v.flatten(), y_v.flatten())))
    # data = torch.from_numpy(usol[:, :, :, 3].reshape(-1,1))
    print("The coodinates have shape {}".format(coords.shape))
    print("The data has shape {}".format(data.shape))
    return coords, data

dataset = Dataset(
    create_data,
    preprocess_kwargs={
        "noise_level": 0.01,
        "normalize_coords": True,
        "normalize_data": True,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 2500},
    device=device,
)

train_dataloader, test_dataloader = get_train_test_loader(dataset, train_test_split=0.8)


network = NN(3, [50, 50, 50, 50], 1)
library = Library2D(poly_order=1)
estimator = Threshold(0.1)
sparsity_scheduler = TrainTestPeriodic()
constraint = LeastSquares()
model = DeepMoD(network, library, estimator, constraint).to(device)
# Defining optimizer
optimizer = torch.optim.Adam(
    model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3
)
train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    sparsity_scheduler,
    log_dir="runs/2DAD/",
    max_iterations=40000,
    delta = 1e-5,
    patience=200,
)

print(model.estimator_coeffs())





