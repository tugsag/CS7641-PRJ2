import numpy as np
import pandas as pd
import mlrose_hiive as ml
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report

SEED = 903454028
np.random.seed(SEED)

def plot(x, y, title, xlabel, ylabel, ticks=None, labels=None, test=None, nested=False, path=None):
    base = 'figures/'
    plt.figure()
    if nested:
        for index, i in enumerate(y):
            plt.plot(x, i, label=labels[index])
    if test is not None:
        plt.plot(x, y, label='train')
        plt.plot(x, test, label='test')
    else:
        plt.plot(x, y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if ticks is not None:
        plt.xticks(ticks=ticks, labels=labels, rotation=90)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig(base + path)
    plt.clf()

def split_arr(arr, n):
    return [arr[i:i+n] for i in range(0, len(arr), n)]

def nn_rhc(train_x, train_y, test_x, test_y):
    restarts = [10, 25, 50, 100]
    # learning_rates = [.001, .01, .1, .5]
    curves = []
    loss = []
    accuracies_train = []
    auc_train = []
    accuracies_test = []
    auc_test = []
    pred = None
    # params = []
    for ind, i in enumerate(restarts):
        # set_accuracy = []
        # set_auc = []
        # set_loss = []
        print(ind/len(restarts), ' progress')
        # for ind, j in enumerate(learning_rates):
            # print(j)
        model = ml.NeuralNetwork(hidden_nodes=[30, 30], activation='relu', algorithm='random_hill_climb', max_iters=1000, bias=True, is_classifier=True,
                        learning_rate=0.1, early_stopping=False, restarts=i, max_attempts=100, random_state=SEED, curve=True)
        model.fit(train_x, train_y)
        loss.append(model.loss)
        curves.append(model.fitness_curve)
        # params.append('restart={}, lr={}'.format(i, j))
        # train
        train_y_pred = model.predict(train_x)
        accuracies_train.append(accuracy_score(train_y, train_y_pred))
        auc_train.append(roc_auc_score(train_y, train_y_pred))

        # test
        pred_y = model.predict(test_x)
        accuracies_test.append(accuracy_score(test_y, pred_y))
        auc = roc_auc_score(test_y, pred_y)
        auc_test.append(auc)
        if max(auc_test) == auc:
            pred = pred_y
        # loss.append(set_loss)
        # accuracies_test.append(set_accuracy)
        # auc_test.append(set_auc)

    # accuracies_test = np.asarray(accuracies_test)
    # auc_test = np.asarray(auc_test)
    # print(auc_test)
    # index = np.argmax(auc_test.mean(axis=1))
    # print(learning_rates[index])
    # accuracies_test = accuracies_test[index]
    # auc_test = auc_test[index]

    curves = np.asarray(curves)
    index = np.argmax(curves.mean(axis=1))
    best_curve = curves[index]
    plot(restarts, loss, 'Loss vs Restarts', 'Restarts', 'Loss', path='nn_rhc_loss.png')
    plot(restarts, accuracies_train, 'Accuracy vs Restarts', 'Restarts', 'Accuracy', test=accuracies_test, path='nn_rhc_accuracy.png')
    plot(restarts, auc_train, 'AUC vs Restarts', 'Restarts', 'AUC Score', test=auc_test, path='nn_rhc_auc.png')
    # plot(params, loss, 'Loss vs Restarts', 'Restarts', 'Loss', ticks=range(len(params)), labels=params)
    # plot(params, accuracies_train, 'Accuracy vs Restarts', 'Restarts', 'Accuracy', ticks=range(len(params)), labels=params, test=accuracies_test)
    # plot(params, auc_train, 'AUC vs Restarts', 'Restarts', 'AUC Score', ticks=range(len(params)), labels=params, test=auc_test)
    plot(range(len(best_curve)), best_curve, 'Best Curve - NN RHC', 'Iterations', 'Fitness', path='nn_rhc_best_curve.png')
    print('best clasification: ')
    print(classification_report(test_y, pred))


def nn_sa(train_x, train_y, test_x, test_y):
    temps = [.01, 0.5, 10, 50]
    learning_rates = [.001, .01, .1, .5]
    curves = []
    loss = []
    accuracies_train = []
    auc_train = []
    accuracies_test = []
    auc_test = []
    pred = None
    for ind, i in enumerate(temps):
        print(ind/len(temps), ' progress')
        for j in learning_rates:
            schedule = ml.ExpDecay(init_temp=i)
            model = ml.NeuralNetwork(hidden_nodes=[10, 10], activation='relu', algorithm='simulated_annealing', max_iters=1000, bias=True, is_classifier=True,
                            learning_rate=j, early_stopping=False, schedule=schedule, max_attempts=100, random_state=SEED, curve=True)
            model.fit(train_x, train_y)
            loss.append(model.loss)
            curves.append(model.fitness_curve)
            # train
            train_y_pred = model.predict(train_x)
            accuracies_train.append(accuracy_score(train_y, train_y_pred))
            auc_train.append(roc_auc_score(train_y, train_y_pred))

            # test
            pred_y = model.predict(test_x)
            accuracies_test.append(accuracy_score(test_y, pred_y))
            auc = roc_auc_score(test_y, pred_y)
            auc_test.append(auc)
            if max(auc_test) == auc:
                pred = pred_y

    accuracies_test = np.asarray(split_arr(accuracies_test, 4))
    auc_test = np.asarray(split_arr(auc_test, 4))
    accuracies_train = np.asarray(split_arr(accuracies_train, 4))
    auc_train = np.asarray(split_arr(auc_train, 4))
    index = np.argmax(auc_test.mean(axis=1))
    best_ind = np.argmax(auc_test)
    print('best lr: ', learning_rates[index])
    accuracies_test = accuracies_test[index]
    auc_test = auc_test[index]
    accuracies_train = accuracies_train[index]
    auc_train = auc_train[index]

    # curves = np.array(split_arr(curves, 4))
    # index = np.argmax(curves.mean(axis=1))
    # best_curve = curves[index]
    loss = np.asarray(split_arr(loss, 4))
    index = np.argmin(loss.mean(axis=1))
    loss = loss[index]


    plot(temps, loss, 'Loss vs Temps', 'Temps', 'Loss', path='nn_sa_loss.png')
    plot(temps, accuracies_train, 'Accuracy vs temps', 'Temps', 'Accuracy', test=accuracies_test, path='nn_sa_accuracy.png')
    plot(temps, auc_train, 'AUC vs Temps', 'Temps', 'AUC Score', test=auc_test, path='nn_sa_auc.png')
    # plot(range(len(best_curve)), best_curve, 'Best Curve - NN RHC', 'Iterations', 'Fitness', path='nn_sa_best_curve.png')
    print('best classification: ')
    print(classification_report(test_y, pred))

def nn_ga(train_x, train_y, test_x, test_y):
    pop_size = [10, 50, 100]
    mutation_prob = [.01, .1, .5]
    # learning_rates = [.01, .1, .5]
    curves = []
    loss = []
    accuracies_train = []
    auc_train = []
    accuracies_test = []
    auc_test = []
    params = []
    pred = None
    for ind, i in enumerate(pop_size):
        print(ind/len(pop_size), ' progress')
        for j in mutation_prob:
            print(j)
            model = ml.NeuralNetwork(hidden_nodes=[30, 30], activation='relu', algorithm='genetic_alg', max_iters=1000, bias=True, is_classifier=True,
                            learning_rate=0.1, early_stopping=False, pop_size=i, mutation_prob=j, max_attempts=100, random_state=SEED, curve=True)
            model.fit(train_x, train_y)
            loss.append(model.loss)
            curves.append(model.fitness_curve)
            params.append('pop_size={}, mutation={}'.format(i, j))
            # train
            train_y_pred = model.predict(train_x)
            accuracies_train.append(accuracy_score(train_y, train_y_pred))
            auc_train.append(roc_auc_score(train_y, train_y_pred))

            # test
            pred_y = model.predict(test_x)
            accuracies_test.append(accuracy_score(test_y, pred_y))
            auc = roc_auc_score(test_y, pred_y)
            auc_test.append(auc)
            if max(auc_test) == auc:
                pred = pred_y

    curves = np.array(curves)
    index = np.argmax(curves.mean(axis=1))
    best_curve = curves[index]
    best_ind = np.argmax(auc_test)
    print(auc_test[best_ind])
    print(params[best_ind])
    plot(range(len(best_curve)), best_curve, 'Best Curve - NN GA', 'Iterations', 'Fitness', path='nn_ga_best_curve.png')
    plot(params, loss, 'Pop Size & Mutation Prob vs Loss', 'Pop Size & Mutation Prob', 'Loss', ticks=range(len(params)), labels=params, path='nn_ga_loss.png')
    plot(params, accuracies_train, 'Pop Size & Mutation Prob vs Accuracy', 'Pop Size & Mutation Prob', 'Accuracy', ticks=range(len(params)), labels=params, test=accuracies_test, path='nn_ga_accuracy.png')
    plot(params, auc_train, 'Pop Size & Mutation Prob vs AUC', 'Pop Size & Mutation Prob', 'AUC', ticks=range(len(params)), labels=params, test=auc_test, path='nn_ga_auc.png')
    print('best classification: ')
    print(classification_report(test_y, pred))

if __name__ == '__main__':
    d = pd.read_csv('gamma/gamma.csv')
    m = {'g': 1, 'h': 0}
    d['class'] = d['class'].map(m)
    split = StratifiedShuffleSplit(n_splits=1, test_size=.33, random_state=SEED)
    for i, j in split.split(d, d['class']):
        train_set = d.loc[i]
        test_set = d.loc[j]
    y_train, y_test = train_set['class'], test_set['class']
    X_train, X_test = train_set.drop('class', axis=1), test_set.drop('class', axis=1)
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    x = input('''Select weight opt algorithm:
                    Hill climb: r,
                    Genetic: g,
                    Annealing: a: ''')
    if x == 'r':
        nn_rhc(X_train, y_train, X_test, y_test)
    elif x == 'g':
        nn_ga(X_train, y_train, X_test, y_test)
    elif x == 'a':
        nn_sa(X_train, y_train, X_test, y_test)
