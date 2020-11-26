from copy import deepcopy

import matplotlib.pyplot as plt
import torch.nn as nn


# A convenience function which we use to copy CNNs
def copy_model(model: nn.Module) -> nn.Module:
    result = deepcopy(model)

    # Copy over the extra metadata we've collected which copy.deepcopy doesn't capture
    if hasattr(model, 'input_activations'):
        result.input_activations = deepcopy(model.input_activations)

    for result_layer, original_layer in zip(result.children(), model.children()):
        if isinstance(result_layer, nn.Conv2d) or isinstance(result_layer, nn.Linear):
            if hasattr(original_layer.weight, 'scale'):
                result_layer.weight.scale = deepcopy(original_layer.weight.scale)
            if hasattr(original_layer, 'activations'):
                result_layer.activations = deepcopy(original_layer.activations)
            if hasattr(original_layer, 'output_scale'):
                result_layer.output_scale = deepcopy(original_layer.output_scale)

    return result


def vis_weights(weights, str):
    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(weights, 20, density=0)

    # add a 'best fit' line
    # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
    # np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    # ax.plot(bins, y, '--')
    ax.set_xlabel('Flattened index')
    ax.set_ylabel('Weights')
    ax.set_title(str)

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

    return n


def vis_weights2(weights):
    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(weights, 200, density=0)

    # add a 'best fit' line
    # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
    # np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    # ax.plot(bins, y, '--')
    ax.set_xlabel('Flattened index')
    ax.set_ylabel('Weights')
    ax.set_title(r'FC1 weights')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
