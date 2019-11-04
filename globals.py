from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

vectorizers = {
    'Count Vectorizer': CountVectorizer,
    'TF-IDF Vectorizer': TfidfVectorizer
}

class_colors = {
    'gcc': 0,
    'icc': 1,
    'clang': 2,
    'L': 0,
    'H': 1
}

models = {
    'Bernoulli Naive Bayes': {
        'name': BernoulliNB,
        'parameters': {
            0: {
                'alpha': 1.0,
                'fit_prior': True
            },
            1: {
                'alpha': 0.0,
                'fit_prior': False
            }
        }
    },
    'Multinomial Naive Bayes': {
        'name': MultinomialNB,
        'parameters': {
            0: {
                'alpha': 1.0,
                'fit_prior': True
            },
            1: {
                'alpha': 0.0,
                'fit_prior': False
            }
        }
    },
    'Gaussian Naive Bayes': {
        'name': GaussianNB,
        'parameters': {
            0: {
                'priors': None,
                'var_smoothing': 0.0000000001
            }
        }
    },
    'Decision Tree': {
        'name': DecisionTreeClassifier,
        'parameters': {
            0: {
                'criterion': 'gini',
                'max_depth': None
            },
            1: {
                'criterion': 'entropy',
                'max_depth': 5
            }
        }
    },
    'Extra-Tree': {
        'name': ExtraTreeClassifier,
        'parameters': {
            0: {
                'criterion': 'gini',
                'max_depth': None
            },
            1: {
                'criterion': 'entropy',
                'max_depth': 5
            }
        }
    },
    'Linear SVC': {
        'name': LinearSVC,
        'parameters': {
            0: {
                'penalty': 'l2',
                'loss': 'squared_hinge'
            },
            1: {
                'penalty': 'l1',
                'loss': 'squared_hinge'
            }
        }
    },
    'SVM with SGD': {
        'name': SGDClassifier,
        'parameters': {
            0: {
                'loss': 'hinge',
                'penalty': 'l2',
                'learning_rate': 'optimal'
            },
            1: {
                'loss': 'huber',
                'penalty': 'l1',
                'learning_rate': 'constant',
                'eta0': 0.0001,
                'early_stopping': True
            },
            2: {
                'loss': 'perceptron',
                'penalty': 'elasticnet',
                'learning_rate': 'optimal',
                'early_stopping': True
            }
        }
    },
    'K Neighbors': {
        'name': KNeighborsClassifier,
        'parameters': {
            0: {
                'weights': 'unfiorm',
                'algorithm': 'kdtrees'
            },
            1: {
                'weights': 'distance',
                'algorithm': 'brute'
            }
        }
    }
}
