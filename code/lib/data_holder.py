import os
import array
import numpy as np
import matplotlib.pyplot as plt

###
### GENERAL UTILITY FUNCTIONS
###

try:
   import cPickle as pickle
except:
   import pickle

def create_cached(cache_file, func):
    """
    Try to unpickle object from file `cache_file` if it exists, otherwise call `func` to create the object and save it to cache.

    You should manually delete cache file when it becomes outdated.
    """

    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, 'rb'))

    obj = func()
    pickle.dump(obj, open(cache_file, 'wb'))
    return obj

###
### COORDINATE CONVERSION
###

"""
ISO 31-11 spherical coordinate system is used everywhere:
Theta: 0 - pi
Phi: 0 - 2*pi
"""

def spherical_to_cartesian(points):
    """
    Converts 2d numpy array of spherical (r, [0;pi] theta, [0;2pi] phi) rows to array 2d with cartesian (x, y, z) rows
    """
    r = points[:,0]

    # 0 - pi
    theta = points[:,1]

    # 0 - 2pi
    phi = points[:,2]

    res = np.zeros(points.shape)
    res[:,0] = r * np.sin(theta) * np.cos(phi)
    res[:,1] = r * np.sin(theta) * np.sin(phi)
    res[:,2] = r * np.cos(theta)
    return res

def map_theta_phi_to_square(theta, phi):
    """
    Map ([0;pi] theta, [0;2pi] phi) points to [0,1]^2 square preserving uniformity of distributions.

    Uniform (theta, phi) distribution on sphere results in a uniform distribution on square.
    Conversion of phi -> x is linear, whereas conversion of theta -> y is non-linear to preserve uniformity.
    """

    assert theta.max() <= np.pi + 1e-3
    assert theta.min() >= -1e-3

    x = phi / (2 * np.pi)
    y = (-np.cos(theta) + 1) / 2
    x[x < 0] += 1
    x[x > 1] -= 1
    y[y < 0] += 1
    y[y > 1] -= 1

    return x, y

###
### POINT ARRAY
###

class PointArray:
    """
    Represents point cloud in 3D space, stored simultaneosly in cartesian and spherical coordinates for convenience.

    Both 'cartesian' and 'spherical' fields are 2d numpy arrays of shape (N, 3)
    spherical holds (r, [0;pi] theta, [0;2pi] phi) rows
    cartesian holds (x, y, z) rows
    """

    """
    Index of R in spherical row
    """
    R = 0

    """
    Index of theta in spherical row
    """
    THETA = 1

    """
    Index of phi in spherical row
    """
    PHI = 2

    def __init__(self):
        self.cartesian = None
        self.spherical = None

    def subarray_for_distance_range(self, r_from, r_to):
        """
        Returns point array with points having r in [r_from, r_to)
        """

        indices = np.where((self.spherical[:, PointArray.R] >= r_from) & (self.spherical[:, PointArray.R] < r_to))[0]
        #print(indices)

        return self.select_points(indices)

    def select_points(self, indices):
        subarr = PointArray()
        subarr.cartesian = self.cartesian[indices,:].copy()
        subarr.spherical = self.spherical[indices,:].copy()
        return subarr

    def get_subsample_if_large(self, size):
        """
        If number of points is larger than `size`, return subsample of size `size`.

        Note: resets np.random.seed and always returns same subsample.
        """

        if self.cartesian.shape[0] <= size:
            return self

        np.random.seed(42)
        indices = np.arange(0, self.cartesian.shape[0])
        np.random.shuffle(indices)
        indices = indices[:size]

        return self.select_points(indices)

###
### DATA HOLDER
###

class DataHolder:
    """
    Provides observations dataset abstraction. (a, b) -> 3d-point cloud

    Supports loading from Maria raw data dumps and conversion to PointArray instances
    """

    def __init__(self, data_path = './korp_stats.bin'):
        self.load_data(data_path)

    def load_data(self, data_path):
        """
        Load data from Maria dump format
        """

        double_size = 8

        n_nums = int(os.path.getsize(data_path) / double_size)
        n_content = 5

        if n_nums % n_content != 0:
            print("Oops, something went wrong with the file size and contents.")

        n_entries = n_nums // n_content

        F = open(data_path, 'rb')
        data = array.array('d')
        data.fromfile(F, n_nums)
        F.close()

        data = np.array(data)
        data = np.reshape(data, (n_entries, n_content))
        self.raw_data = data

        # NOTE: somehow there is ligand with index -1, shift all indices by one for convenience
        self.raw_data[:,1] += 1

        self._process_raw_data()

    def _process_raw_data(self):
        """
        Calculate some stats to ensure data makes sense
        """

        self.max_row = self.raw_data.max(axis=0)
        self.min_row = self.raw_data.min(axis=0)

        self.num_proteins = int(self.max_row[0]) + 1
        self.num_ligands = int(self.max_row[1]) + 1
        self.min_distance = self.min_row[2]
        self.max_distance = self.max_row[2]

    def describe(self):
        """
        Print number of objects and componentwise minimums and maximums
        """

        print('Dataset contains {} proteins x {} ligands, total {} points; distance range {:.2f}-{:.2f} Å'
            .format(
                self.num_proteins,
                self.num_ligands,
                len(self.raw_data),
                self.min_distance,
                self.max_distance))
        print('Min row: {}'.format(self.min_row))
        print('Max row: {}'.format(self.max_row))

    def get_raw_spherical_points_for_pair(self, a, b):
        """
        Returns raw interaction data for protein 'a' and ligand 'b'

        Result is 2d numpy array with shape (N, 3)
        Rows have format (r, [0;pi] theta, [0;2pi] phi)
        """

        assert a < self.num_proteins
        assert b < self.num_ligands
        return self.raw_data[np.where((self.raw_data[:, 0] == a) & (self.raw_data[:, 1] == b))][:, 2:]

    def get_point_array(self, a, b):
        """
        Return PointArray object for protein 'a' and ligand 'b'
        """

        arr = PointArray()
        arr.spherical = self.get_raw_spherical_points_for_pair(a, b)
        arr.cartesian = spherical_to_cartesian(arr.spherical)
        return arr

###
### VISUALIZATIONS AND PLOTTING
###

def scatter_plot_point_array(point_array, ax=None):
    x, y = map_theta_phi_to_square(point_array.spherical[:, PointArray.THETA], point_array.spherical[:, PointArray.PHI])

    x = x * 2 * np.pi
    y = y * np.pi

    if ax is None:
        ax = plt.gca()

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, np.pi)
    ax.set_aspect(2)
    #plt.text(0.1, 0.85, '({},{}]'.format(d_min, d_max), color='fuchsia', fontsize=20)
    ax.scatter(x, y, s=1000.0 / len(x))

def imshow_plot_density(density_grid, description=None, fontsize=20, ax=None):
    dg = density_grid.copy()
    #dg = dg.T

    if ax is None:
        ax = plt.gca()

    # first index is X, second index is Y
    # plot orientation is usual: (0, 0) at bottom left
    ax.imshow(dg, cmap='Greys', extent=[0, 2*np.pi, 0,np.pi], aspect=2, origin='lower')
    if description is not None:
        ax.text(0.1, np.pi*0.92, description, color='fuchsia', fontsize=fontsize)
