import math

def prev(k, cells):
    neighbors = {}
    for idx, cell in enumerate(cells):
        # Only earlier cells can be neighbors
        prev_cells = cells[:idx]
        # Compute distances to earlier cells
        sorted_neighbors = sorted(prev_cells, key=lambda p: math.hypot(cell[0]-p[0], cell[1]-p[1]))
        neighbors[cell] = sorted_neighbors[:k]
    return neighbors


def full(k, cells):
    result = {}
    for cell in cells:
        # Include cell itself in the sorted list.
        sorted_neighbors = sorted(cells, key=lambda p: math.hypot(cell[0] - p[0], cell[1] - p[1]))
        result[cell] = sorted_neighbors[:k]
    return result
