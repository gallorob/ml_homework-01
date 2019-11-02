from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from globals import *


def prepare_vectors(dataset, vect_mode, input, target):
    """
    Prepare the vectors for train and test, given the dataset and the vectorization mode
    :param dataset: the dataset (dictionary)
    :param vect_mode: the vectorization mode
    :param input: the input feature
    :param target: the target feature
    :return: X and Y for train and test
    """
    vect = vectorizers[vect_mode]()
    xs = vect.fit_transform(dataset[input])
    ys = dataset[target]

    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.3, random_state=10)

    return x_train, y_train, x_test, y_test


def evaluate_model(x_train, x_test, y_train, y_test, model, toarray):
    """
    Evaluate the given model
    :param x_train: the training inputs
    :param x_test: the testing inputs
    :param y_train: the training labels
    :param y_test: the testing labels
    :param model: the model
    :param toarray: whether to cast to an array (from a sparse matrix)
    :return: the confusion matrix and the classification report
    """
    if toarray:
        x_train = x_train.toarray()
        x_test = x_test.toarray()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    conf_mat = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    return conf_mat, class_report

