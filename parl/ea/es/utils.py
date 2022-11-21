import scipy.stats


def centered_ranks(x):
    '''
    Rank the rewards and normalize them to [0,1) then to (-0.5, 0.5) with sum=0.0
    '''
    # rank from 0 to len(x)-1
    rank = scipy.stats.rankdata(x)-1
    norm_rank = rank / (len(rank) - 1)
    norm_rank -= 0.5
    return norm_rank
