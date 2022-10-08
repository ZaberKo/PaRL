#%%
from ray.rllib.utils import merge_dicts

d1={
    'a':{
        'a1':22,
        'a2':33
    }
}

d2={
    'a':{
        "b1":3333,
        "b2":2
    }
}

merge_dicts(d1,d2)
# %%
d1.update(d2)
# %%
