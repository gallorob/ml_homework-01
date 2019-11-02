import ast
import random

import matplotlib.pyplot as plt

from custom_classes import Sample


def read_json(filename, max_n=None):
    """
    Read a jsonl file and return the corresponding map
    :param filename: the file name
    :return: a list of samples
    """
    samples = []
    with open(filename) as f:
        if max_n is not None:
            lines = list(f)
            while max_n > 0:
                values = ast.literal_eval(random.choice(lines))
                samples.append(Sample(values))
                max_n -= 1
        else:
            for line in f:
                values = ast.literal_eval(line)
                samples.append(Sample(values))
    return samples


def samples_to_dataset(samples, split_instructions=False):
    dataset = {
        "instructions": [],
        "opt": [],
        "compiler": []
    }
    for sample in samples:
        if split_instructions:
            dataset['instructions'].append(' '.join(sample.no_parameters_instructions()))
        else:
            dataset['instructions'].append(sample.instructions)
        dataset["opt"].append(sample.opt)
        dataset["compiler"].append(sample.compiler)
    return dataset


def plotting(dataset):
    xs = dataset['n_jumps']
    ys = dataset['length_instructions']
    area = 1
    colors = dataset['compiler']
    plt.scatter(xs, ys, s=area, c=colors, alpha=0.5)
    plt.title('MSI & LI wrt Opt')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
