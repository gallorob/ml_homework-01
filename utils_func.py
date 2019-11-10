import ast
import random
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import plot_tree

from custom_classes import Sample
from globals import class_colors


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


def save_results(model_name, hparams, vectorizer, pred, split_data, conf_mat, class_report, counter):
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
        results.write(str.format('Result at {date}-{counter}:\n\
        \n\tModel: {model} with parameters {params};\
        \n\tprediction: {pred};\
        \n\tVectorizer: {vect}\
        \n\tFeature extraction method: {method}\
        \nConfusion matrix:\
        \n{conf_mat}\nClassification report:\n{class_report}\n\n',
                                 date=datetime.now().strftime('%Y-%m-%d'),
                                 counter=str.zfill(str(counter), 3),
                                 model=model_name,
                                 params=hparams,
                                 pred=pred,
                                 vect=vectorizer,
                                 method='No parameters' if split_data else 'No feature extraction',
                                 conf_mat=conf_mat,
                                 class_report=class_report))


def set_object_attributes(object, attributes):
    """
    Set the object's attributes dynamically
    :param object: the object
    :param attributes: the attributes
    """
    for k in attributes:
        setattr(object, k, attributes[k])


def plot_and_save(x_test, y_pred, model, title, counter):
    """
    Plot the samples and the model if it's a tree, then save
    :param x_test: inputs
    :param y_pred: predictions
    :param model: the model
    :param title: graph title
    :param counter: image counter
    """
    # plot model if tree
    if title.find('Tree') != -1:
        tree_fig = plt.figure(figsize=(8, 8), dpi=80)
        plot_tree(model, filled=True)
        plt.title(title)
        plt.show()
        tree_fig.savefig('pictures\\tree_plot_{counter}.png'.format(counter=str.zfill(str(counter), 3)))
    scatter_fig = plt.figure(figsize=(8, 8), dpi=80)
    area = 1
    dimensions = 2
    pca = PCA(n_components=dimensions).fit(x_test.todense())
    data = pca.transform(x_test.todense())
    colors = []
    for y in y_pred:
        colors.append(class_colors[y])
    plt.scatter(data[:, 0], data[:, 1], s=area, c=colors)
    plt.title(title)
    plt.show()
    scatter_fig.savefig('pictures\\results_{counter}.png'.format(counter=str.zfill(str(counter), 3)))
