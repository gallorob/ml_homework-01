import numpy as np

from ml_tools import *
from utils_func import *


def run_fit_and_predictions(dataset, vect_mode, input, target, model_type, model_params, split_data, counter=0):
    x_train, y_train, x_test, y_test = prepare_vectors(dataset=dataset,
                                                       vect_mode=vect_mode,
                                                       input=input,
                                                       target=target)
    model = models[model_type]['name']()
    set_object_attributes(object=model,
                          attributes=model_params)
    conf_mat, class_report, y_pred = evaluate_model(x_train=x_train,
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
                 class_report=class_report,
                 counter=counter)

    title = model_type + '_' + str(model_params) + '_' + ('Instruction Only' if split_data else 'Full Instruction') + '_predicting_' + target
    plot_and_save(x_test, y_pred, model, title, counter)


def plot_dataset(dataset):
    predicting = ['opt', 'compiler']
    for pred in predicting:
        x_train, y_train, x_test, y_test = prepare_vectors(dataset=dataset,
                                                           vect_mode='TF-IDF Vectorizer',
                                                           input='instructions',
                                                           target=pred)
        plot_and_save(x_train, y_train, None, 'Train dataset', 102)
        plot_and_save(x_test, y_test, None, 'Test dataset', 103)


if __name__ == '__main__':
    np.random.seed(0)  # fix the random seed for replicability

    split_data = True  # setting this to False may lead to 'array too big' errors

    print('Started loading dataset...')

    samples = read_json(filename='train_dataset.jsonl')
    dataset = samples_to_dataset(samples=samples,
                                 split_instructions=split_data)

    print('Dataset loaded and transformed')

    # Plot the dataset
    plot_dataset(dataset)

    # Evaluate all given models

    predicting = ['opt', 'compiler']
    run_models = models.keys()
    i = 0

    for vector in vectorizers.keys():
        for model in run_models:
            for params in models[model]['parameters']:
                for pred in predicting:
                    print('Evaluating {model} with {params} using vectorization {vector} and features type {feature}; predicting {pred}...'
                          .format(model=model,
                                  params=str(models[model]['parameters'][params]),
                                  vector=vector,
                                  feature=('Instruction Only' if split_data else 'Full Instruction'),
                                  pred=pred))
                    run_fit_and_predictions(dataset=dataset,
                                            vect_mode=vector,
                                            input='instructions',
                                            target=pred,
                                            model_type=model,
                                            model_params=models[model]['parameters'][params],
                                            split_data=split_data,
                                            counter=i)
                    i += 1

    # BLIND PREDICTIONS

    print('Started blind predictions...')

    # Prepare vectorizer
    vect = vectorizers['TF-IDF Vectorizer']()
    _ = vect.fit_transform(dataset['instructions'])

    print('Fitting KNN...')
    target = 'opt'
    model_type = 'K Neighbors'
    param = models[model_type]['parameters'][0]
    knn = models[model_type]['name']()
    set_object_attributes(object=knn,
                          attributes=param)

    x_train, y_train, _, _ = prepare_vectors(dataset=dataset,
                                             vect_mode='TF-IDF Vectorizer',
                                             input='instructions',
                                             target=target)
    knn.fit(x_train, y_train)

    print('Fitting Decision Tree...')
    target = 'compiler'
    model_type = 'Decision Tree'
    param = models[model_type]['parameters'][0]
    dectree = models[model_type]['name']()
    set_object_attributes(object=dectree,
                          attributes=param)

    x_train, y_train, _, _ = prepare_vectors(dataset=dataset,
                                             vect_mode='TF-IDF Vectorizer',
                                             input='instructions',
                                             target=target)
    dectree.fit(x_train, y_train)

    print('Started loading blind dataset...')
    # clear memory
    samples = None
    dataset = None
    # load dataset and transform
    blind_samples = read_json(filename='test_dataset_blind.jsonl')
    blind_dataset = []
    for sample in blind_samples:
            blind_dataset.append(' '.join(sample.no_parameters_instructions()))
    xs = vect.transform(blind_dataset)
    print('Blind data loaded, making predictions...')

    # OPT
    print('Optimizer predictions...')
    blind_opt_pred = knn.predict(xs)

    # COMPILER
    print('Compiler predictions...')
    blind_compiler_pred = dectree.predict(xs)

    # Write csv
    print('Writing to .csv the predictions...')
    matricula = '1890251'
    with open('{0}.csv'.format(matricula), 'w') as csvfile:
        for i in range(len(blind_opt_pred)):
            csvfile.write(blind_compiler_pred[i] + ',' + blind_opt_pred[i] + '\n')
