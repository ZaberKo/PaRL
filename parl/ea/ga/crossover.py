import numpy as np
from ray.rllib.utils.typing import ModelWeights

def crossover_inplace(gene1: ModelWeights, gene2: ModelWeights):
    for W1, W2 in zip(gene1.values(), gene2.values()):

        if len(W1.shape) == 2:  # Weights no bias
            num_variables = W1.shape[0]
            # Crossover opertation [Indexed by row]
            try:
                num_cross_overs = np.random.randint(
                    0, int(num_variables * 0.3))  # Number of Cross overs
            except:
                num_cross_overs = 1
            for i in range(num_cross_overs):
                # Choose which gene to receive the perturbation
                receiver_choice = np.random.rand()
                if receiver_choice < 0.5:
                    ind_cr = np.random.randint(0, W1.shape[0])  #
                    W1[ind_cr, :] = W2[ind_cr, :]
                else:
                    ind_cr = np.random.randint(0, W1.shape[0])  #
                    W2[ind_cr, :] = W1[ind_cr, :]

        elif len(W1.shape) == 1:  # Bias
            if np.random.rand() < 0.8:
                continue  # Crossover here with low frequency
            num_variables = W1.shape[0]
            # Crossover opertation [Indexed by row]
            # num_cross_overs = np.random.randint(0, int(num_variables * 0.05))  # Crossover number
            for i in range(1):
                # Choose which gene to receive the perturbation
                receiver_choice = np.random.rand()
                if receiver_choice < 0.5:
                    ind_cr = np.random.randint(0, W1.shape[0])  #
                    W1[ind_cr] = W2[ind_cr]
                else:
                    ind_cr = np.random.randint(0, W1.shape[0])  #
                    W2[ind_cr] = W1[ind_cr]
