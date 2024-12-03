"""Coinflip with weights."""

import math

import numpy as np


def n_weighted_coinflip(weights):
    """Perform a weighted coinflip.

    Parameters
    ----------
    weights : list
        List of weights.

    Returns
    -------
    str
        The index of the weights.

    Raises
    ------
    ValueError
        If the sum of the weights is not 1.
    """
    if not np.isclose(np.sum(weights), 1):
        raise ValueError(f"The sum is not 1: {np.sum(weights)}")

    random = np.random.rand()
    calc = 0
    for i, weight in enumerate(weights):
        calc = calc + weight
        if random < calc:
            return format(i, "0" + str(math.floor(math.log2(len(weights)))) + "b")
