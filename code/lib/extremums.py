import numpy as np
import matplotlib.pyplot as plt

def kernel_extremums_2d(pdf, kernel_size=10, min_ratio_to_max_pdf=0.2, min_pdf=0):
    # kernel size
    ok = None

    # small noise to avoid multi-maximums
    np.random.seed(42)
    pdf = pdf + np.random.randn(*pdf.shape) * (pdf.min() + 1e-6) * 0.1

    def convolution(matrix):
        res = matrix
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                res = np.maximum(res, np.roll(matrix, (i, j), axis=(0, 1)))
        return res

    assert kernel_size >= 1
    conv_res = convolution(pdf)
    for i in range(kernel_size - 1):
        conv_res = convolution(conv_res)
    #print('conv res', conv_res)

    extremums = np.argwhere((pdf == conv_res) & (pdf > max(min_ratio_to_max_pdf * pdf.max(), min_pdf)))

    filtered = []
    for index in range(extremums.shape[0]):
        if extremums[index, 0] < pdf.shape[0] * 0.05 or extremums[index, 0] > pdf.shape[0] * 0.95:
            continue
        filtered.append(extremums[index, :])
    return np.array(filtered)

def plot_extremums(extremums, pdf, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.scatter(
        (extremums[:, 1] + 0.5) / pdf.shape[1] * (2 * np.pi),
        (extremums[:, 0] + 0.5) / pdf.shape[0] * np.pi, s=8.0, color='red')
