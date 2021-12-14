import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(42)  # Seed for reproducibility

# Lorenz model
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

# Generate training data

delta_t = 0.001
t_train = np.arange(0, 100, delta_t)
t_train_range = (t_train[0], t_train[-1])
x0_train = [-8, 8, 27] # initial point x0,y0,z0
x_train = solve_ivp(lorenz, t_train_range, x0_train, t_eval=t_train, rtol = 1e-12, method = 'LSODA', atol = 1e-12).y.T
x_dot_train_measured = np.array(
    [lorenz(0, x_train[i]) for i in range(t_train.size)]
)

# find identified data with different noise levels
poly_order = 5
threshold = 0.05
noise_levels = [1e-4, 1e-3, 1e-2, 1e-1, 1]
models = []
t_sim = np.arange(0, 20, delta_t)
x_sim = []
for eps in noise_levels:
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_order),
    )
    model.fit(
        x_train,
        t=delta_t,
        x_dot=x_dot_train_measured
        + np.random.normal(scale=eps, size=x_train.shape),
        quiet=True,
    )
    models.append(model)
    x_sim.append(model.simulate(x_train[0], t_sim))

# show model with lowest noise level:
print(f'Identified model with noise level {noise_levels[0]} is:')
models[0].print()

# Look at the error over multiple thresholds
thresholds = [i/100 for i in range(1,100,5)]
models_t = []
t_sim = np.arange(0, 20, delta_t)
x_simt = []
eps = 0.001
for t in thresholds:
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=t),
        feature_library=ps.PolynomialLibrary(degree=poly_order),
    )
    model.fit(
        x_train,
        t=delta_t,
        x_dot=x_dot_train_measured
        + np.random.normal(scale=eps, size=x_train.shape),
        quiet=True,
    )
    models_t.append(model)
    x_simt.append(model.simulate(x_train[0], t_sim))    

# Plot results

# Plot true data without noise
fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(131, projection="3d")
ax.plot(
    x_train[: t_sim.size, 0],
    x_train[: t_sim.size, 1],
    x_train[: t_sim.size, 2],
)
plt.title("Training data no noise")
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")

# plot identified model data with small noise
model_idx = 0
ax = fig.add_subplot(132, projection="3d")
ax.plot(x_sim[model_idx][:, 0], x_sim[model_idx][:, 1], x_sim[model_idx][:, 2])
plt.title(f"SINDy identified system, noise level={noise_levels[model_idx]}")
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")

# plot identified data with higher noise
model_idx = 3
ax = fig.add_subplot(133, projection="3d")
ax.plot(x_sim[model_idx][:, 0], x_sim[model_idx][:, 1], x_sim[model_idx][:, 2])
plt.title(f"SINDy identified system, noise level={noise_levels[model_idx]}")
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.savefig('./graphs/lorenz.png')

# plot x vs t for both training and simulated data
fig = plt.figure(figsize=(12, 5))
model_idx = 0
ax = fig.add_subplot(221)
ax.plot(t_sim, x_train[: t_sim.size, 0], "r", label = 'training data')
ax.plot(t_sim, x_sim[model_idx][:, 0], "b--", label = 'simulated data')
plt.title(f"noise level={noise_levels[model_idx]}")
plt.ylabel("x")

# plot y vs t for both training and simulated data
ax = fig.add_subplot(223)
ax.plot(t_sim, x_train[: t_sim.size, 1], "r", label = 'training data')
ax.plot(t_sim, x_sim[model_idx][:, 1], "b--", label = 'simulated data')
plt.xlabel("time")
plt.ylabel("y")

model_idx = 3
# plot x vs t for both training and simulated data
ax = fig.add_subplot(222)
ax.plot(t_sim, x_train[: t_sim.size, 0], "r", label = 'training data')
ax.plot(t_sim, x_sim[model_idx][:, 0], "b--", label = 'simulated data')
plt.title(f"noise level={noise_levels[model_idx]}")
plt.ylabel("x")

# plot x vs t for both training and simulated data
ax = fig.add_subplot(224)
ax.plot(t_sim, x_train[: t_sim.size, 1], "r", label = 'training data')
ax.plot(t_sim, x_sim[model_idx][:, 1], "k--", label = 'simulated data')
plt.xlabel("time")
plt.ylabel("y")
plt.savefig('./graphs/lorenz2.png')

# errors (RMSE vs noise):
rmse_average_noise = []
for model_idx in range(len(noise_levels)):
	rmse_average_noise.append(np.average(np.sqrt((x_sim[model_idx][:, 0]-x_train[: t_sim.size, 0])**2+(x_sim[model_idx][:, 1]-x_train[: t_sim.size, 1])**2),axis = 0))
fig,ax = plt.subplots(figsize=(12, 8))
ax.plot(noise_levels, rmse_average_noise, '-ok')
ax.set_title('RMSE vs noise')
ax.set_ylabel('RMSE in both x and y')
ax.set_xlabel('Noise levels')
ax.set_yscale('log')
plt.savefig('./graphs/lorenz3.png')

# errors (RMSE vs threshold):
rmse_average_t = []
for model_idx in range(len(thresholds)):
	rmse_average_t.append(np.average(np.sqrt((x_simt[model_idx][:, 0]-x_train[: t_sim.size, 0])**2+(x_simt[model_idx][:, 1]-x_train[: t_sim.size, 1])**2),axis = 0))
fig,ax = plt.subplots(figsize=(12, 8))
ax.plot(thresholds, rmse_average_t, '-ok')
ax.set_title('RMSE vs threshold level')
ax.set_ylabel('RMSE in both x and y')
ax.set_xlabel('Threshold levels')
ax.set_yscale('log')
plt.savefig('./graphs/lorenz4.png')
# plt.show()    