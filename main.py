from sklearn.naive_bayes import MultinomialNB

from ml_tools import *
from utils_func import *

if __name__ == '__main__':
    samples = read_json('train_dataset.jsonl')
    dataset = samples_to_dataset(samples, True)
    x_train, y_train, x_test, y_test = prepare_count_vectorizer(dataset, 'instructions', 'compiler')
    model = MultinomialNB()
    conf_mat, class_report = evaluate_model(x_train, y_train, x_test, y_test, model)

    print(conf_mat)
    print(class_report)

