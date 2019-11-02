import ast
import random
from datetime import datetime

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
    """
    Convert a list of samples into a dictionary dataset
    :param samples: list of samples
    :param split_instructions: whether to consider the single operation or the whole instruction
    :return: the dataset
    """
    dataset = {
        "instructions": [],
        "opt": [],
        "compiler": []
    }
    for sample in samples:
        if split_instructions:
            dataset['instructions'].append(' '.join(sample.no_parameters_instructions()))
        else:
            dataset['instructions'].append(sample.instructions_as_string())
        dataset["opt"].append(sample.opt)
        dataset["compiler"].append(sample.compiler)
    return dataset


def save_results(model_name, hparams, vectorizer, pred, split_data, conf_mat, class_report):
    """
    Save the predictions result in a file
    :param model_name: the name of the model
    :param hparams: the hyper parameters
    :param pred: the target prediction
    :param vectorizer: the vectorizer type
    :param conf_mat: the confusion matrix
    :param class_report: the classification report
    """
    with open('results.log', 'a') as results:
        results.write(str.format('Result at {date}:\n\
        \n\tModel: {model} with parameters {params};\
        \n\tprediction: {pred};\
        \n\tVectorizer: {vect}\
        \n\tFeature extraction method: {method}\
        \nConfusion matrix:\
        \n{conf_mat}\nClassification report:\n{class_report}\n\n',
                                 date=datetime.now().strftime('%Y-%m-%d'),
                                 model=model_name,
                                 params=hparams,
                                 pred=pred,
                                 vect=vectorizer,
                                 method='No parameters' if split_data else 'No feature extraction',
                                 conf_mat=conf_mat,
                                 class_report=class_report))


def set_object_attributes(object, attributes):
    for k in attributes:
        setattr(object, k, attributes[k])


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
