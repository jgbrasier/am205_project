import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(42)  # Seed for reproducibility

# Linear, damped harmonic oscillator:
def linear_damped_SHO(t, x):
	"""x' = -0.1x+2y
	y' = -2x-0.1y
	"""
	return [-0.1 * x[0] + 2 * x[1], -2 * x[0] - 0.1 * x[1]]

# Generate training data
delta_t = 0.01
t_train = np.arange(0, 30, delta_t)
t_train_range = (t_train[0], t_train[-1])
x0_train = [1, 0] # initial condition
x_train = solve_ivp(linear_damped_SHO, t_train_range, x0_train, t_eval=t_train, rtol = 1e-12, method = 'LSODA', atol = 1e-12).y.T

poly_order = 5 # looking at space of polynomials up to 5th order
threshold = 0.05 # selecting by default a threshold equal t0 0.5

# Build the model
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    feature_library=ps.PolynomialLibrary(degree=poly_order),
)


# Look at the results over multiple noise levels
x_dot_train_measured = np.array(
    [linear_damped_SHO(0, x_train[i]) for i in range(t_train.size)]
)
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

# plot actual training data without noise
fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(131)
ax.plot(
    x_train[: t_sim.size, 0],
    x_train[: t_sim.size, 1]
)
plt.title("Training data no noise")
ax.set(xlabel="$x$", ylabel="$y$")

# plot identified model data with small noise
model_idx = 0
ax = fig.add_subplot(132)
ax.plot(x_sim[model_idx][:, 0], x_sim[model_idx][:, 1])
plt.title(f"SINDy identified system, Noise level={noise_levels[model_idx]}")
ax.set(xlabel="$x$", ylabel="$y$")

# plot identified data with higher noise
model_idx = 3
ax = fig.add_subplot(133)
ax.plot(x_sim[model_idx][:, 0], x_sim[model_idx][:, 1])
plt.title(f"SINDy identified system, Noise level={noise_levels[model_idx]}")
ax.set(xlabel="$x$", ylabel="$y$")
plt.savefig('./graphs/vibration.png')

fig = plt.figure(figsize=(12, 5))
model_idx = 0
ax = fig.add_subplot(221)
ax.plot(t_sim, x_train[: t_sim.size, 0], "r", label = 'Training data')
ax.plot(t_sim, x_sim[model_idx][:, 0], "b--", label = 'Simulated data')
plt.title(f"x, y vs t for Noise level={noise_levels[model_idx]}")
plt.ylabel("x")
plt.legend()

ax = fig.add_subplot(223)
ax.plot(t_sim, x_train[: t_sim.size, 1], "r",label = 'Training data')
ax.plot(t_sim, x_sim[model_idx][:, 1], "b--", label = 'Simulated data')
plt.xlabel("time")
plt.ylabel("y")
plt.legend()

model_idx = 3
ax = fig.add_subplot(222)
ax.plot(t_sim, x_train[: t_sim.size, 0], "r",label = 'Training data')
ax.plot(t_sim, x_sim[model_idx][:, 0], "b--", label = 'Simulated data')
plt.title(f"x, y vs t for Noise level={noise_levels[model_idx]}")
plt.ylabel("x")
plt.legend()

ax = fig.add_subplot(224)
ax.plot(t_sim, x_train[: t_sim.size, 1], "r",label = 'Training data')
ax.plot(t_sim, x_sim[model_idx][:, 1], "b--", label = 'Simulated data')
plt.xlabel("time")
plt.ylabel("y")
plt.legend()
plt.savefig('./graphs/vibration2.png')

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
plt.savefig('./graphs/vibration3.png')
# fig.tight_layout()

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
plt.savefig('./graphs/vibration4.png')

# plt.show()
