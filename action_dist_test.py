# %%
from policy.action_dist import SquashedGaussian2
import numpy as np

import torch

net = torch.nn.Linear(25, 10)

input = np.random.rand(10)
dist = SquashedGaussian2(input, net, low=-0.01, high=0.01)

# %%
action = dist.sample()
action
# %%
np.set_printoptions(precision=10, suppress=True)
raw_act = dist._unsquash(action)
print(raw_act)
act = dist._squash(raw_act)
print(act)
# %%
dist.logp(action)
# %%


def logp(dist, x):
    # Unsquash values (from [low,high] to ]-inf,inf[)
    unsquashed_values = dist._unsquash(x)
    # Get log prob of unsquashed values from our Normal.
    log_prob_gaussian = dist.dist.log_prob(unsquashed_values)
    # For safety reasons, clamp somehow, only then sum up.
    log_prob_gaussian = torch.clamp(log_prob_gaussian, -100, 100)
    log_prob_gaussian = torch.sum(log_prob_gaussian, dim=-1)
    # Get log-prob for squashed Gaussian.
    unsquashed_values_tanhd = torch.tanh(unsquashed_values)
    log_prob = log_prob_gaussian - torch.sum(
        torch.log(1 - unsquashed_values_tanhd ** 2), dim=-1
    )
    return log_prob


cnt = 0
for i in range(100):
    action = dist.sample()
    p1 = dist.logp(action)
    p2 = logp(dist, action)
    print(p1.item(),p2.item(),(p1-p2).item())


# %%
