import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import entropy


def max_uncertainty(x, y):
    x_height = len(x)
    x_width = len(x[0])

    assert x.shape == y.shape

    uncertainty = np.zeros(shape=x.shape)

    for i, j in product(range(x_height), range(x_width)):
        p1, p2 = x[i, j], y[i, j]
        if p1 + p2 >= 1:
            uncertainty[i, j] = 0
        else:
            p3 = 1 - p1 - p2
            uncertainty[i, j] = 1 - np.max([p1, p2, p3])

    return uncertainty


def max_margin(x, y):
    x_height = len(x)
    x_width = len(x[0])

    assert x.shape == y.shape

    margin = np.zeros(shape=x.shape)

    for i, j in product(range(x_height), range(x_width)):
        p1, p2 = x[i, j], y[i, j]
        if p1 + p2 >= 1:
            margin[i, j] = 0
        else:
            p3 = 1 - p1 - p2
            part = np.partition(-np.asarray([p1, p2, p3]), 1)
            margin[i, j] = 1 + part[0] - part[1]

    return margin


def ternary_entropy(x, y):
    x_height = len(x)
    x_width = len(x[0])

    assert x.shape == y.shape

    entr = np.zeros(shape=x.shape)

    for i, j in product(range(x_height), range(x_width)):
        p1, p2 = x[i, j], y[i, j]
        if p1 + p2 >= 1:
            entr[i, j] = 0
        else:
            p3 = 1 - p1 - p2
            entr[i, j] = entropy([p1, p2, p3])

    return entr


def contour_plot_2D(
        x, y, vals, level_res, cmap=plt.cm.Greys,
        title='', xlabel='', ylabel='',
        export_path=None, filename=None
):
    with plt.style.context('seaborn-whitegrid'):
        fig = plt.figure(figsize=(15, 15))
        # plotting kde
        plt.contourf(
            x, y, vals,
            levels=np.linspace(0, np.max(vals), level_res),
            cmap=cmap
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if export_path is None and filename is None:
            plt.show()

        if export_path is not None and filename is not None:
            plt.savefig(os.path.join(export_path, filename))

        plt.close('all')


# uncertainty measures
res = 1000
level_res = 500
x, y = np.meshgrid(
    np.linspace(0, 1, res),
    np.linspace(0, 1, res)
)

'''for i in range(len(x)):
    for j in range(len(x[0])):
        if x[i, j] + y[i, j] >= 1:
            x[i, j] = 0
            y[i, j] = 0'''

contour_plot_2D(
    x, y, max_uncertainty(x, y), level_res,
    title='Classification uncertainty', xlabel='p1', ylabel='p2',
    export_path=os.getcwd(), filename='uncertainty.png'
)
contour_plot_2D(
    x, y, max_margin(x, y), level_res,
    title='Classification margin', xlabel='p1', ylabel='p2',
    export_path=os.getcwd(), filename='margin.png'
)
contour_plot_2D(
    x, y, ternary_entropy(x, y), level_res,
    title='Classification entropy', xlabel='p1', ylabel='p2',
    export_path=os.getcwd(), filename='entropy.png'
)

exit()

uncertainty_measure_vals = [max_uncertainty(x, y), max_margin(x, y), ternary_entropy(x, y)]
plt_titles = ['Classifier uncertainty', 'Classifier margin', 'Classifier entropy']

with plt.style.context('seaborn-white'):
    fig = plt.figure(figsize=(30, 10))
    for idx in range(3):
        plt.subplot(1, 3, idx+1)
        plt.contourf(
            x, y, uncertainty_measure_vals[idx],
            levels=np.linspace(0, np.max(uncertainty_measure_vals[idx]), level_res),
            cmap=plt.cm.Greys
        )
        plt.title(plt_titles[idx])
        plt.xlabel('p1')
        plt.ylabel('p2')
    plt.show()

