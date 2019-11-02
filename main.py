import numpy as np

from ml_tools import *
from utils_func import *


def run_fit_and_predictions(dataset, vect_mode, input, target, model_type, model_params, split_data):
    x_train, y_train, x_test, y_test = prepare_vectors(dataset=dataset,
                                                       vect_mode=vect_mode,
                                                       input=input,
                                                       target=target)
    model = models[model_type]['name']()
    set_object_attributes(object=model,
                          attributes=model_params)
    conf_mat, class_report = evaluate_model(x_train=x_train,
                                            y_train=y_train,
                                            x_test=x_test,
                                            y_test=y_test,
                                            model=model,
                                            toarray=split_data)

    save_results(model_name=model_type,
                 hparams=model_params,
                 vectorizer=vect_mode,
                 pred=target,
                 split_data=split_data,
                 conf_mat=conf_mat,
                 class_report=class_report)

    # fig = plt.figure()
    # plot_tree(model, filled=True)
    # plt.show()
    # fig.savefig('pictures\\tree01_limit_opt.png')


if __name__ == '__main__':
    np.random.seed(0)  # fix the random seed for reproducibility

    split_data = True
    vectorizer = 'TF-IDF Vectorizer'
    model = 'Decision Tree'
    predicting = 'compiler'

    samples = read_json(filename='train_dataset.jsonl')  # , max_n=1000)
    dataset = samples_to_dataset(samples=samples,
                                 split_instructions=split_data)
    params = models[model]['parameters'][1]
    run_fit_and_predictions(dataset=dataset,
                            vect_mode=vectorizer,
                            input='instructions',
                            target=predicting,
                            model_type=model,
                            model_params=params,
                            split_data=split_data)
