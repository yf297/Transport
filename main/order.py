import math
import itertools

def max_min(grid_size):
    cells = list(itertools.product(range(grid_size), range(grid_size)))
    center = (grid_size - 1) / 2
    start = min(cells, key=lambda p: (p[0] - center)**2 + (p[1] - center)**2)

    #ordered = [start]
    #cells.remove(start)

    # Max-min selection
    '''while cells:
        best = None
        best_min_dist = -1
        for candidate in cells:
            min_dist = min(math.hypot(candidate[0]-p[0], candidate[1]-p[1]) for p in ordered)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best = candidate
        ordered.append(best)
        cells.remove(best)'''
    return cells
