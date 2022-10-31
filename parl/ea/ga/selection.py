import numpy as np

def selection_tournament(index_rank, num_offsprings, tournament_size):
    """ Conduct tournament selection by random selection (instead of by fitness)

        Parameters:
                  index_rank (list): Ranking encoded as net_indexes
                  num_offsprings (int): Number of offsprings to generate
                  tournament_size (int): Size of tournament

        Returns:
                  offsprings (list): List of offsprings returned as a list of net indices
    """

    total_choices = len(index_rank)
    offsprings = []
    for _ in range(num_offsprings):
        winner = np.min(np.random.randint(
            total_choices, size=tournament_size))
        offsprings.append(index_rank[winner])

    offsprings = list(set(offsprings))  # Find unique offsprings
    if len(offsprings) % 2 != 0:  # Number of offsprings should be even
        offsprings.append(index_rank[winner])
    return offsprings
