import argparse
import sys
import datetime

from lib.data_holder import *
from lib.models import *
from lib.extremums import *

###
### GLOBALS
###

CACHE_DIR = 'cache'

data_holder = None

def lazy_data_holder():
    global data_holder
    if data_holder is None:
        data_holder = DataHolder()

    return data_holder

###
### PLOT POINT CLOUD
###

def plot_point_cloud():
    """Plot point cloud for protein-ligand pair."""

    options_parser = argparse.ArgumentParser(add_help=True)
    options_parser.add_argument('--protein', help='protein index', default=0, type=int)
    options_parser.add_argument('--ligand', help='ligand index', default=5, type=int)
    options = options_parser.parse_args()

    cache_file_name = '{}/{}_{}.points'.format(CACHE_DIR, options.protein, options.ligand)
    arr = create_cached(cache_file_name, lambda: lazy_data_holder().get_point_array(options.protein, options.ligand))
    print('Have {} points for pair ({}, {})'.format(len(arr.spherical), options.protein, options.ligand))

    subarr = arr.subarray_for_distance_range(4.5, 5.5)
    print(len(subarr.spherical))

    scatter_plot_point_array(subarr)
    plt.savefig('test.png')

###
### PLOT DENSITY
###

from matplotlib.backends.backend_pdf import PdfPages

def plot_density_with_params(protein, ligand, r_from, r_to):
    cache_file_name = '{}/{}_{}.points'.format(CACHE_DIR, protein, ligand)
    arr = create_cached(cache_file_name, lambda: lazy_data_holder().get_point_array(protein, ligand))
    print('Have {} points for pair ({}, {})'.format(len(arr.spherical), protein, ligand))

    # 0, 5, 4.75-5.25 ok
    # 0, 6, 4.75-5.25 ok
    # 1, 1, 4.25-5.75 ok

    pdf_output = PdfPages('plot_{}_{}.pdf'.format(protein, ligand))

    #subarr = arr.subarray_for_distance_range(4.75, 5.25)
    subarr = arr.subarray_for_distance_range(r_from, r_to)
    print(len(subarr.spherical))

    scatter_plot_point_array(subarr)
    plt.xlabel(r'$\varphi$')
    plt.ylabel(r'$\theta$')
    plt.savefig('plots/points.png')
    plt.title('Source points')
    pdf_output.savefig(plt.gcf())
    plt.close()

    def make_model_density_plot(model, output_file, title):
        dist_point_array = get_point_array_for_distance(5)
        density = model.get_density_for_points(dist_point_array)
        density = density.reshape(100, 100)

        imshow_plot_density(density)

        extremums = kernel_extremums_2d(density)
        plot_extremums(extremums, density)

        plt.xlabel(r'$\varphi$')
        plt.ylabel(r'$\theta$')
        plt.title(title)
        plt.savefig(output_file)
        pdf_output.savefig(plt.gcf())
        plt.close()


    if True:
        kernel_density_model = KernelDensityModel(subarr)
        make_model_density_plot(kernel_density_model, 'plots/density_kernel.png', 'Parzen')

    if True:
        histogram_density_model = HistogramDensityModel(subarr, r_steps=3, theta_steps=25, phi_steps=25)
        make_model_density_plot(histogram_density_model, 'plots/density_histogram.png', 'Histogram')

    if True:
        gm_density_model = GaussianMixtureDensityModel(subarr)
        make_model_density_plot(gm_density_model, 'plots/density_gmm.png', 'Gaussian mixture')

    def compare_models(models):
        dist_point_array = union_point_arrays([get_point_array_for_distance(5), get_point_array_for_distance(10), get_point_array_for_distance(15)])
        volumes = get_point_array_approx_volumes(dist_point_array, n_r=3, n_phi=100, n_theta=100)

        densities = [model.get_density_for_points(dist_point_array) for model in models]

        def get_average_diff_for_pair(i, j):
            # MAPE
            #return (np.abs(densities[i] - densities[j]) / (np.minimum(densities[i], densities[j]) + 1e-6) * volumes).sum() / volumes.sum()

            # MSE
            return ((densities[i] - densities[j]) ** 2 * volumes).sum() / volumes.sum()

        for i in range(len(models)):
            for j in range(len(models)):
                print('pair {} - {}: {}'.format(i, j, get_average_diff_for_pair(i, j)))

    compare_models([kernel_density_model, histogram_density_model, gm_density_model])
    pdf_output.close()

    if False:
        def make_model_density_plots_multiple(model, output_file):
            dist_point_array = get_point_array_for_distance(5)
            densities = model.get_density_for_points(dist_point_array)

            for index, density in enumerate(densities):
                imshow_plot_density(density.reshape(100, 100))
                plt.savefig(output_file.format(index))
                plt.close()


        N = 10
        model = NeuralNetworkModel(subarr, num_iterations=N)
        for t in range(300):
            make_model_density_plots_multiple(model, 'plots/nn/density_nn_{}_' + str(model.total_iterations) + '.png')
            model.optimize_for(N)

def plot_density():
    """Plot point cloud for protein-ligand pair."""

    options_parser = argparse.ArgumentParser(add_help=True)
    options_parser.add_argument('--protein', help='protein index', default=1, type=int)
    options_parser.add_argument('--ligand', help='ligand index', default=1, type=int)
    options = options_parser.parse_args()

    plot_density_with_params(options.protein, options.ligand, 4.25, 5.75)

def plot_density_preset():
    plot_density_with_params(0, 5, 4.75, 5.25)
    plot_density_with_params(0, 0, 4.75, 5.25)
    plot_density_with_params(1, 1, 4.25, 5.75)

###
### MOD CHOOSER
###

def mod_chooser_main(available_modes):
    available_mode_names = [
        mode.__name__ for mode in available_modes
    ]

    is_known_mode = len(sys.argv) >= 2 and sys.argv[1] in available_mode_names
    need_help = len(sys.argv) >= 2 and sys.argv[1] == '--help'

    if not need_help and not is_known_mode and len(sys.argv) >= 2:
        print('Unknown mode: "{}"\n'.format(sys.argv[1], available_mode_names))

    if not is_known_mode or need_help:
        print('Usage: {} <mode> <options>'.format(sys.argv[0]))
        print('')
        print('Available modes:')
        for mode in available_modes:
            print('{} - {}'.format(mode.__name__, mode.__doc__))
        sys.exit(1)

    mode = sys.argv[1]
    mode_func = available_modes[available_mode_names.index(mode)]
    del sys.argv[1]
    mode_func()

###
### MAIN
###

def main():
    available_modes = [
        plot_point_cloud,
        plot_density,
        plot_density_preset
    ]

    mod_chooser_main(available_modes)

if __name__ == '__main__':
    main()

