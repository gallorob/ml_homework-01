from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


def prepare_count_vectorizer(dataset, input, target):
    vect = TfidfVectorizer()
    xs = vect.fit_transform(dataset[input])
    ys = dataset[target]

    x_train, y_train, x_test, y_test = train_test_split(xs, ys, test_size=0.3, random_state=10)

    return x_train, y_train, x_test, y_test


def evaluate_model(x_train, x_test, y_train, y_test, model):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    conf_mat = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    return conf_mat, class_report

