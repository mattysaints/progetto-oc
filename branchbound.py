import numpy as np
from tsp import TSP


def bb_tsp(tsp: TSP):
    u = np.inf
    stack = [({}, {})]

    while stack:
        subproblem = stack.pop(0)

        # TODO finire
    pass
