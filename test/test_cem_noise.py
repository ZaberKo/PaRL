#%%
import numpy as np
import matplotlib.pyplot as plt


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
    noise_decay_coeff=0.9995,
    test_num=10000
)

# %%
