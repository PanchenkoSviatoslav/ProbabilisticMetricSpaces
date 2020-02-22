from lib.data_holder import *
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

###
### 3D spherical grid generation
###

# Our goal: we want a 3d numpy array (meshgrid)

# or, better yet, handle everything in cartesian from the start?
# the only complexity then is finding closest pivot point
# easier to integrate & handle overall
# more coarse plots
# 10^6 points cloud minimum
# whereas spherical could be like 20 * 100 * 100 = 2*10^5 and look better

# Want common interface:
# meshgrid (cartesian) -> density estimates -> closest pivots -> density plot

class SpaceGrid:
    def __init__(self):
        # 4d array: (ix, iy, iz) -> (x, y, z) row
        self.point_grid = None
        # 3d array: (ix, iy, iz) -> volume of a cell
        self.volume_grid = None

def get_cartesian_grid():
    r_max = 20

    X, Y, Z = np.meshgrid(np.linspace(-r_max, r_max, 100), np.linspace(-r_max, r_max, 100), np.linspace(-r_max, r_max, 100), indexing='ijk')

# Side note - do we really need 3d arrays?
# they have a lot of points and are slow to calculate

def get_point_array_for_distance(r):
    theta_linspace = np.arccos(np.linspace(1, -1, 100))
    phi_linspace = np.linspace(0, 2 * np.pi, 100)
    theta, phi = np.meshgrid(theta_linspace, phi_linspace, indexing='ij')

    point_array = PointArray()
    point_array.spherical = np.hstack([
        np.ones((theta.size, 1)) * r,
        theta.ravel().reshape(theta.size, 1),
        phi.ravel().reshape(phi.size, 1)
    ])
    point_array.cartesian = spherical_to_cartesian(point_array.spherical)

    return point_array

def get_point_array_approx_volumes(point_array, n_r, n_theta, n_phi):
    r_w = 20 / n_r
    theta_w = np.pi / n_theta
    phi_w = 2 * np.pi / n_phi
    return point_array.spherical[:, 0] ** 2 * np.sin(point_array.spherical[:, 1]) * phi_w * theta_w * r_w

def union_point_arrays(arrays):
    res = PointArray()
    res.spherical = np.vstack([arr.spherical for arr in arrays])
    res.cartesian = np.vstack([arr.cartesian for arr in arrays])
    return res

###
### Kernel density model
###

class KernelDensityModel:
    MAX_POINTS_FOR_FIT = 30000

    def __init__(self, point_array, bandwidth=0.5):
        self.model = KernelDensity(
            bandwidth=bandwidth, metric='euclidean', kernel='gaussian', algorithm='auto', rtol=0.1)
        self.model.fit(point_array.get_subsample_if_large(self.MAX_POINTS_FOR_FIT).cartesian)

    """
    def init_old(self, spherical_points, bandwidth=0.5, description=None):
        self.description = description
        self.spherical_points = spherical_points
        self.kde = KernelDensity(
            bandwidth=bandwidth, metric='euclidean', kernel='gaussian', algorithm='auto', rtol=0.1)

        max_points_for_fit = 10000
        points = spherical_to_cartesian(spherical_points)
        np.random.seed(42)
        np.random.shuffle(points)
        points = points[:max_points_for_fit,:]

        self.kde.fit(points)
    """

    def get_density_for_points(self, point_array):
        return np.exp(self.model.score_samples(point_array.cartesian))

    """
    def get_density_grid_for_distance(self, r):
        #y_linspace = np.linspace(0, np.pi, 100)
        y_linspace = np.arccos(np.linspace(1, -1, 100))

        X, Y = np.meshgrid(np.linspace(0, 2 * np.pi, 100), y_linspace, indexing='ij')

        #print('zero-zero (2pi, pi)', X[0, 0], Y[0, 0])

        probe_points = spherical_to_cartesian(np.hstack(
            [np.ones((X.size, 1)) * r, Y.ravel().reshape(X.size, 1), X.ravel().reshape(Y.size, 1)]))
        prob = np.exp(self.kde.score_samples(probe_points))

        return prob.reshape(X.shape)

    def plot_density_for_distance(self, r, fontsize=20, no_axis=False):
        plot_density(self.get_density_grid_for_distance(r),
                     no_axis=no_axis, description=self.description, fontsize=fontsize)
        plt.text(0.1, np.pi*0.82, 'r={}Å'.format(r), color='fuchsia', fontsize=fontsize)

        if no_axis:
            plt.gca().set_yticklabels([])
            plt.gca().set_xticklabels([])

    def plot_point_cloud_for_range(self, d_min, d_max):
        plot_for_distance_range(self.spherical_points, d_min, d_max)

    def plot_point_cloud(self, r):
        self.plot_point_cloud_for_range(r - 0.5, r + 0.5)

    def comparative_plot(self, r):
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        self.plot_density_for_distance(r)
        plt.subplot(1, 2, 2)
        self.plot_point_cloud(r)
        plt.show()
    """

###
### Histogram density
###

class HistogramDensityModel:
    MAX_POINTS_FOR_FIT = 900000

    def __init__(self, point_array, r_steps=6, theta_steps=100, phi_steps=100):
        spherical_points = point_array.get_subsample_if_large(self.MAX_POINTS_FOR_FIT).spherical.copy()

        self.r_min = 1.5
        self.r_max = 19.5
        self.r_steps = r_steps
        self.r_w = (self.r_max - self.r_min) / self.r_steps

        # theta 0 - pi, phi 0-2pi

        # (r, theta, phi)

        self.theta_steps = theta_steps
        self.theta_w = np.pi / self.theta_steps

        self.phi_steps = phi_steps
        self.phi_w = 2 * np.pi / self.phi_steps

        self.hist = np.zeros((self.r_steps, self.theta_steps, self.phi_steps))

        # n points in volume v
        # p * v = n / N
        # p = n / N / v

        indices = self.map_indices_batch(spherical_points)
        good_indices = np.where((indices[:, PointArray.R] >= 0) & (indices[:, PointArray.R] < self.hist.shape[0]))[0]
        indices = indices[good_indices, :]
        spherical_points = spherical_points[good_indices, :]

        volumes = spherical_points[:, 0] ** 2 * np.sin(spherical_points[:, 1]) * self.phi_w * self.theta_w * self.r_w

        # hack to avoid division by zero around poles
        # (discard points that were unsuccessful enough to map there)
        volumes[volumes < 0.01] = 10000

        volumes = 1.0 / volumes

        #a = np.array([0])
        #a[np.array([0, 0])] += 1
        #np.add.at(a.ravel(), [0, 0], 1)
        #print(a)

        #self.hist[indices] += volumes
        #self.hist.ravel()[np.ravel_multi_index(indices.T, self.hist.shape)] += volumes.ravel()

        if False:
            for i in range(len(spherical_points)):
                #r = p[0]
                #theta = p[1]
                #phi = p[2]

                #r_i, theta_i, phi_i = self.map_indices(r, theta, phi)

                r_i, theta_i, phi_i = indices[i, :]

                #if not (r_i >= 0 and r_i < self.hist.shape[0]):
                #    continue

                # todo: calculate volume
                # dV = r^2 sin(theta) dphi dtheta dr
                # approximate value
                #volume = r ** 2 * np.sin(theta) * self.phi_w * self.theta_w * self.r_w
                #volume = volumes[i]

                self.hist[r_i, theta_i, phi_i] += volumes[i]
        else:
            np.add.at(self.hist.ravel(), np.ravel_multi_index(indices.T, self.hist.shape), volumes.ravel())

        self.hist /= (len(spherical_points))

    def map_indices_batch(self, spherical_points):
        res = np.zeros(spherical_points.shape, dtype=np.int64)
        res[:, 0] = np.floor((spherical_points[:, 0] - self.r_min) / self.r_w).astype(np.int64)
        res[:, 1] = np.floor(spherical_points[:, 1] / self.theta_w).astype(np.int64) % self.theta_steps
        res[:, 2] = np.floor(spherical_points[:, 2] / self.phi_w).astype(np.int64) % self.phi_steps
        return res

    def map_indices(self, r, theta, phi):
        r_i = int(np.floor((r - self.r_min) / self.r_w))
        theta_i = int(np.floor(theta / self.theta_w)) % self.theta_steps
        phi_i = int(np.floor(phi / self.phi_w)) % self.phi_steps
        return r_i, theta_i, phi_i

    def get_density_for_points(self, point_array):
        #res = np.zeros(point_array.spherical.shape[0])
        indices = self.map_indices_batch(point_array.spherical)

        #print(indices[:, 0].max())

        #res = self.hist[indices]
        return np.take(self.hist, np.ravel_multi_index(indices.T, self.hist.shape))
        #return res

        #for i in range(len(res)):
        #    r, theta, phi = point_array.spherical[i, :]
        #    r_i, theta_i, phi_i = self.map_indices(r, theta, phi)
        #    res[i] = self.hist[r_i, theta_i, phi_i]
        #return res

    """
    def get_density_grid_for_distance(self, r):
        y_linspace = np.arccos(np.linspace(1, -1, 100))

        X, Y = np.meshgrid(np.linspace(0, 2 * np.pi, 100), y_linspace, indexing='ij')
        res = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                theta = X[i, j]
                phi = Y[i, j]

                r_i, phi_i, theta_i = self.map_indices(r, phi, theta)

                res[i, j] = self.hist[r_i, phi_i, theta_i]

        return res
    """

###
### Gaussian mixture model
###

class GaussianMixtureDensityModel:
    MAX_POINTS_FOR_FIT = 12000

    def __init__(self, point_array):
        self.model = GaussianMixture(n_components=20, covariance_type='full', max_iter=1000)
        self.model.fit(point_array.get_subsample_if_large(self.MAX_POINTS_FOR_FIT).cartesian)

    def get_density_for_points(self, point_array):
        return np.exp(self.model.score_samples(point_array.cartesian))

import torch

# attempt to implement
# Neural Networks for Density Estimation : SIC (Smooth Interpolation of the Cumulative)
class NeuralNetworkModelBad:
    class Network:
        def __init__(self):
            # N is batch size; D_in is input dimension;
            # H is hidden dimension; D_out is output dimension.
            self.D_in = 3
            self.H = 1
            self.D_out = 1

            self.w1 = torch.randn(self.D_in, self.H, requires_grad=True)
            self.w2 = torch.randn(self.H, self.D_out, requires_grad=True)

            with torch.no_grad():
                self.w1 /= np.sqrt(self.H)
                self.w2 /= np.sqrt(self.H)

        def forward(self, x):
            return x.mm(self.w1).clamp(min=0).mm(self.w2)


    MAX_POINTS_FOR_FIT = 1000

    def __init__(self, point_array):
        self.fit(point_array.get_subsample_if_large(self.MAX_POINTS_FOR_FIT).cartesian)

    def fit(self, points):
        N = points.shape[0]

        self.model = self.Network()

        #x = torch.randn(N, D_in, device=device)
        x = torch.from_numpy(points.astype(np.float32))
        y = torch.randn(N, self.model.D_out)

        print('build train')
        for i in range(N):
            s = 0
            for j in range(N):
                if (points[j] <= points[i]).all():
                    s += 1
            y[i, 0] = s / N
        print('done')

        n_random_points = N
        random_points = torch.randn(n_random_points, self.model.D_in)
        random_points *= 20.0

        random_points_delta = random_points + torch.ones(n_random_points, self.model.D_in) * 0.1

        for t in range(2000):
            learning_rate = 1e-5
            y_pred = self.model.forward(x)

            y_pred_rp = self.model.forward(random_points)
            y_pred_rp_delta = self.model.forward(random_points_delta)

            leq_zero = y_pred_rp - y_pred_rp_delta

            # Compute and print loss. Loss is a Tensor of shape (), and loss.item()
            # is a Python number giving its value.
            loss = (y_pred - y).pow(2).sum() + (leq_zero.clamp(min=0) * leq_zero.pow(2)).sum() * 5

            if t % 5 == 0:
                print(t, loss.item())

            loss.backward()
            with torch.no_grad():
                self.model.w1 -= learning_rate * self.model.w1.grad
                self.model.w2 -= learning_rate * self.model.w2.grad

                self.model.w1.grad.zero_()
                self.model.w2.grad.zero_()

        #self.kde = MLPRegressor(n_components=20, covariance_type='full')
        #    self.kde.fit(point_array.get_subsample_if_large(self.MAX_POINTS_FOR_FIT).cartesian)

    def get_density_for_points(self, point_array):
        def calc_at(x):
            x = torch.from_numpy(x.astype(np.float32))
            y_pred = self.model.forward(x).detach().numpy()
            return y_pred

        p = point_array.cartesian
        d = 0.2

        #return calc_at(p)

        ix = np.zeros(p.shape)
        iy = ix.copy()
        iz = ix.copy()
        ix[:,0] = d
        iy[:,1] = d
        iz[:,2] = d

        def get_dx(p):
            return (calc_at(p + ix) - calc_at(p - ix)) / (2 * d)

        def get_dxdy(p):
            return (get_dx(p + iy) - get_dx(p - iy)) / (2 * d)

        def get_dxdydz(p):
            return (get_dxdy(p + iz) - get_dxdy(p - iz)) / (2 * d)

        return calc_at(p)

        #return np.ones(point_array.spherical.shape[0])
        #return np.exp(self.kde.score_samples(point_array.cartesian))


# https://arxiv.org/pdf/1804.05316.pdf
# CDF2PDF
class NeuralNetworkModelBad2:
    class Network:
        def __init__(self):
            # N is batch size; D_in is input dimension;
            # H is hidden dimension; D_out is output dimension.
            self.D_in = 3
            self.H = 100
            self.D_out = 1

            self.w1 = torch.randn(self.D_in, self.H, requires_grad=True)
            self.b1 = torch.randn(self.H, requires_grad=True)

            self.w2 = torch.randn(self.H, self.D_out, requires_grad=True)
            self.b2 = torch.randn(self.D_out, requires_grad=True)

            with torch.no_grad():
                self.w1 /= np.sqrt(self.H) * 10
                self.b1 /= np.sqrt(self.H) * 10

                self.w2 /= np.sqrt(self.H) * 10
                self.b2 /= np.sqrt(self.H) * 10

        def forward(self, x):
            res = x.mm(torch.exp(self.w1)) + self.b1
            res = torch.sigmoid(res)
            res = res.mm(torch.exp(self.w2)) + self.b2

            return res

        def parameters(self):
            return self.w1, self.w2, self.b1, self.b2


    MAX_POINTS_FOR_FIT = 10000

    def __init__(self, point_array, num_iterations):
        self.fit(self.get_input(point_array.get_subsample_if_large(self.MAX_POINTS_FOR_FIT)), num_iterations)

    def get_input(self, point_array):
        return point_array.cartesian

    def fit(self, points, num_iterations):
        N = points.shape[0]

        self.model = self.Network()

        #x = torch.randn(N, D_in, device=device)
        self.x_train = torch.from_numpy(points.astype(np.float32))
        self.y_train = torch.randn(N, self.model.D_out)

        print('build train')
        for i in range(N):
            s = (points <= points[i]).all(axis=1).sum()
            self.y_train[i, 0] = s / N
            #print(s)
        print('done')

        learning_rate = 1e-2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.total_iterations = 0
        self.optimize_for(num_iterations)

    def optimize_for(self, num_iterations):
        for t in range(num_iterations):
            self.total_iterations += 1
            self.optimizer.zero_grad()
            y_pred = self.model.forward(self.x_train)

            # Compute and print loss. Loss is a Tensor of shape (), and loss.item()
            # is a Python number giving its value.
            loss = (y_pred - self.y_train).pow(2).sum() / len(y_pred)

            if self.total_iterations % 5 == 0:
                print(self.total_iterations, loss.item())

            loss.backward()
            self.optimizer.step()

        #self.kde = MLPRegressor(n_components=20, covariance_type='full')
        #    self.kde.fit(point_array.get_subsample_if_large(self.MAX_POINTS_FOR_FIT).cartesian)

    def get_density_for_points(self, point_array):
        def calc_at(x):
            x = torch.from_numpy(x.astype(np.float32))
            y_pred = self.model.forward(x).detach().numpy()
            return y_pred

        p = self.get_input(point_array)
        d = 0.2

        #return calc_at(p)

        ix = np.zeros(p.shape)
        iy = ix.copy()
        iz = ix.copy()
        ix[:,0] = d
        iy[:,1] = d
        iz[:,2] = d

        if False:
            def get_dx(p):
                return (calc_at(p + ix) - calc_at(p - ix)) / (2 * d)

            def get_dxdy(p):
                return (get_dx(p + iy) - get_dx(p - iy)) / (2 * d)

            def get_dxdydz(p):
                return (get_dxdy(p + iz) - get_dxdy(p - iz)) / (2 * d)
        else:
            def get_dx(p):
                return (calc_at(p + ix) - calc_at(p)) / d

            def get_dxdy(p):
                return (get_dx(p + iy) - get_dx(p)) / d

            def get_dxdydz(p):
                return (get_dxdy(p + iz) - get_dxdy(p)) / d

        return [get_dxdydz(p)]
        #return [calc_at(p), get_dx(p), get_dxdy(p), get_dxdydz(p)]

        #return np.ones(point_array.spherical.shape[0])
        #return np.exp(self.kde.score_samples(point_array.cartesian))

class NeuralNetworkModel:
    class Network:
        def __init__(self):
            # N is batch size; D_in is input dimension;
            # H is hidden dimension; D_out is output dimension.
            self.D_in = 3
            self.H = 100
            self.D_out = 1

            self.w1 = torch.randn(self.D_in, self.H, requires_grad=True)
            self.b1 = torch.randn(self.H, requires_grad=True)

            self.w2 = torch.randn(self.H, self.D_out, requires_grad=True)
            self.b2 = torch.randn(self.D_out, requires_grad=True)

            with torch.no_grad():
                self.w1 /= np.sqrt(self.H) * 10
                self.b1 /= np.sqrt(self.H) * 10

                self.w2 /= np.sqrt(self.H) * 10
                self.b2 /= np.sqrt(self.H) * 10

        def forward(self, x):
            res = x.mm(self.w1) + self.b1
            res = res.clamp(min=0)
            res = res.mm(self.w2) + self.b2

            return res

        def parameters(self):
            return self.w1, self.w2, self.b1, self.b2


    MAX_POINTS_FOR_FIT = 10000

    def __init__(self, point_array, num_iterations):
        self.inner_model = HistogramDensityModel(point_array)
        self.fit(point_array.get_subsample_if_large(self.MAX_POINTS_FOR_FIT), num_iterations)

    def get_input(self, point_array):
        return point_array.spherical

    def fit(self, point_array, num_iterations):
        points = self.get_input(point_array)
        N = points.shape[0]

        self.model = self.Network()

        #x = torch.randn(N, D_in, device=device)
        self.x_train = torch.from_numpy(points.astype(np.float32))
        self.y_train = torch.from_numpy(self.inner_model.get_density_for_points(point_array).astype(np.float32))

        learning_rate = 1e-2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.total_iterations = 0
        self.optimize_for(num_iterations)

    def optimize_for(self, num_iterations):
        for t in range(num_iterations):
            self.total_iterations += 1
            self.optimizer.zero_grad()
            y_pred = self.model.forward(self.x_train)

            # Compute and print loss. Loss is a Tensor of shape (), and loss.item()
            # is a Python number giving its value.
            loss = (y_pred - self.y_train).pow(2).sum() / len(y_pred)

            if self.total_iterations % 5 == 0:
                print(self.total_iterations, loss.item())

            loss.backward()
            self.optimizer.step()

        #self.kde = MLPRegressor(n_components=20, covariance_type='full')
        #    self.kde.fit(point_array.get_subsample_if_large(self.MAX_POINTS_FOR_FIT).cartesian)

    def get_density_for_points(self, point_array):
        def calc_at(x):
            x = torch.from_numpy(x.astype(np.float32))
            y_pred = self.model.forward(x).detach().numpy()
            return y_pred

        p = self.get_input(point_array)
        return [calc_at(p)]
        #return [calc_at(p), get_dx(p), get_dxdy(p), get_dxdydz(p)]

        #return np.ones(point_array.spherical.shape[0])
        #return np.exp(self.kde.score_samples(point_array.cartesian))
