#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def plot(
        noise_init=1e-3,
        noise_end=1e-5,
        noise_decay_coeff=0.95,
        test_num=1000
):

    noise = noise_init
    result = [noise]
    for i in range(test_num):
        noise = noise_decay_coeff*noise + (1-noise_decay_coeff)*noise_end
        result.append(noise)

    plt.figure(dpi=300, figsize=(9, 3))
    ax=plt.axes()
    plt.plot(result)
    ax.set_ylim(noise_end, noise_init)
    plt.show()


# %%
plot(
    noise_init=1e-3,
    noise_end=1e-5,
    noise_decay_coeff=0.995,
    test_num=1000
)

# %%
def es_weights(pop_size=10):
    _ws = np.log(pop_size+0.5)-np.log(np.arange(1, pop_size+1))
    ws = _ws/_ws.sum()
    print(ws)

    plt.figure(dpi=300, figsize=(9, 3))
    ax=plt.axes()
    plt.plot(ws)
    ax.set_ylim(0, ws[0]*1.2)
    plt.show()
#%%
es_weights(pop_size=50)
# %%
def cem_weights(num_elites=10):
    _ws = np.log(1+num_elites)/np.arange(1, num_elites+1)
    ws = _ws/_ws.sum()
    print(ws)

    plt.figure(dpi=300, figsize=(9, 3))
    ax=plt.axes()
    plt.plot(ws)
    ax.set_ylim(0, ws[0]*1.2)
    plt.show()

# %%
cem_weights(50)
# CEM的weights下降更快一些, 但仍保留了long tail的权重, 使其不为0
# 因此可能更适合elites?
# %%
