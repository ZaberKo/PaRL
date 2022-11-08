#%%
import numpy as np

import scipy.stats

def centered_ranks(x):
    '''
    Rank the rewards and normalize them.
    '''
    # rank from 0 to len(x)-1
    rank = scipy.stats.rankdata(x)-1
    norm_rank = rank / (len(rank) - 1)
    norm_rank -= 0.5
    return norm_rank

#%%
a=np.random.randint(100,size=(10))
a
# %%
centered_ranks(a).sum()
# %%
