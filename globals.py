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
    'Extra-Tree': ExtraTreeClassifier,
    'Linear SVC': LinearSVC,
    'SVM with SGD': SGDClassifier,
    'K Neighbors': {
        'name': KNeighborsClassifier,
        'parameters': {
            'weights': ['uniform', 'distance'],
            'algorithm': ['kdtree', 'brute']
        }
    }
}
