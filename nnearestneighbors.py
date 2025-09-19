import numpy as np

def n_nearest_neighbors(px, py, x, y, n=1, exclude_points=None, max_distance=None, min_distance=None):
    """
    Find the n nearest neighbors to a given point (px, py) 
    in a dataset defined by x and y arrays, with options to:
      - exclude multiple points by coordinates
      - restrict neighbors to a distance range [min_distance, max_distance]

    Parameters
    ----------
    px, py : float
        The query point coordinates.
    x, y : array-like, shape (m,)
        Arrays of x and y coordinates for m points.
    n : int
        Number of nearest neighbors to return.
    exclude_points : list of tuples [(ex1, ey1), (ex2, ey2), ...], optional
        Points to exclude from the search.
    max_distance : float, optional
        Maximum distance cutoff for neighbors. Points farther than this are ignored.
    min_distance : float, optional
        Minimum distance cutoff for neighbors. Points closer than this are ignored.

    Returns
    -------
    neighbors : ndarray, shape (k, 2)
        The neighbors found (up to n, but fewer if cutoffs apply).
    indices : ndarray, shape (k,)
        The indices of the neighbors in the dataset.
    distances : ndarray, shape (k,)
        The distances to the neighbors.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Compute distances
    distances = np.sqrt((x - px)**2 + (y - py)**2)

    # Exclude multiple points by coordinates
    if exclude_points is not None:
        exclude_points = np.atleast_2d(exclude_points)  # ensures shape (k,2)
        for ex, ey in exclude_points:
            mask = (x == ex) & (y == ey)
            distances[mask] = np.inf

    # Apply maximum distance cutoff
    if max_distance is not None:
        distances[distances > max_distance] = np.inf

    # Apply minimum distance cutoff
    if min_distance is not None:
        distances[distances < min_distance] = np.inf

    # Handle case where fewer than n valid neighbors exist
    valid = np.isfinite(distances)
    n = min(n, np.sum(valid))

    if n == 0:
        return np.array([]), np.array([]), np.array([])

    # Get indices of n smallest distances
    idx = np.argpartition(distances, n)[:n]

    # Sort them by distance
    idx = idx[np.argsort(distances[idx])]

    neighbors = np.column_stack((x[idx], y[idx]))

    return neighbors, idx, distances[idx]
