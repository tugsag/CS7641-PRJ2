import numpy as np
import pandas as pd
import mlrose_hiive as ml
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, learning_curve
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, roc_curve

SEED = 903454028
N_JOBS = 16
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

def nn_rhc(train_x, train_y, test_x, test_y):
    restarts = [10, 25, 50]
    # learning_rates = [.001, .01, .1, .5]
    best_restart = 0
    best_auc = 0
    times = []
    for ind, i in enumerate(restarts):
        start = time.time()
        print(ind/len(restarts), ' progress')
        model = ml.NeuralNetwork(hidden_nodes=[30, 30], activation='relu', algorithm='random_hill_climb', max_iters=1000, bias=True, is_classifier=True,
                        learning_rate=0.1, early_stopping=False, restarts=i, max_attempts=100, random_state=SEED, curve=True)
        model.fit(train_x, train_y)
        # test
        pred_y = model.predict(test_x)
        auc = roc_auc_score(test_y, pred_y)
        times.append(time.time() - start)
        if auc > best_auc:
            best_restart = i
        print(auc, ' ', i)

    print('times for reach restart: ', list(zip(restarts, times)))
    print('total time for all: ', sum(times))
    # Learning Curve
    print('starting plots')
    model = ml.NeuralNetwork(hidden_nodes=[30, 30], activation='relu', algorithm='random_hill_climb', max_iters=1000, bias=True, is_classifier=True,
                     learning_rate=0.1, early_stopping=False, restarts=best_restart, max_attempts=100, random_state=SEED)
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, train_x, train_y, n_jobs=N_JOBS,
                       train_sizes=np.linspace(0.1, 1.0, 5), scoring='f1_weighted', verbose=5, random_state=SEED, return_times=True)
    plt.title('RHC Learning')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.grid()
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig('figures/nn_rhc_learning.png')
    plt.clf()

    # ROC Curve
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    y_probs = model.predicted_probs
    fpr, tpr, _ = roc_curve(test_y, y_probs)
    auc = roc_auc_score(test_y, pred_y)
    plt.plot(fpr, tpr, label='ROC Curve, area={}'.format(auc))
    plt.legend(loc='lower right')
    plt.savefig('figures/nn_rhc_roc.png')
    plt.clf()

    print('best clasification: ')
    print(classification_report(test_y, pred_y))

def nn_sa(train_x, train_y, test_x, test_y):
    temps = [.01, 0.5, 10, 50]
    learning_rates = [.001, .01, .1, .5]
    times = []
    params = []
    best_temp = 0
    best_lr = 0
    best_auc = 0
    for ind, i in enumerate(temps):
        print(ind/len(temps), ' progress')
        for j in learning_rates:
            start = time.time()
            schedule = ml.ExpDecay(init_temp=i)
            model = ml.NeuralNetwork(hidden_nodes=[30, 30], activation='relu', algorithm='simulated_annealing', max_iters=1000, bias=True, is_classifier=True,
                            learning_rate=j, early_stopping=False, schedule=schedule, max_attempts=100, random_state=SEED, curve=True)
            model.fit(train_x, train_y)
            # test
            pred_y = model.predict(test_x)
            auc = roc_auc_score(test_y, pred_y)
            print(auc, ' ', i, ' ', j)
            times.append(time.time() - start)
            if auc > best_auc:
                best_temp = i
                best_lr = j
            params.append('temp={}, lr={}'.format(i, j))

    print('times for params: ', list(zip(params, times)))
    print('total time: ', sum(times))
    # Learning Curve
    print('starting plots')
    schedule = ml.ExpDecay(init_temp=best_temp)
    model = ml.NeuralNetwork(hidden_nodes=[30, 30], activation='relu', algorithm='simulated_annealing', max_iters=1000, bias=True, is_classifier=True,
                            learning_rate=best_lr, early_stopping=False, schedule=schedule, max_attempts=100, random_state=SEED, curve=True)
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, train_x, train_y, n_jobs=N_JOBS,
                       train_sizes=np.linspace(0.1, 1.0, 5), scoring='f1_weighted', verbose=5, random_state=SEED, return_times=True)
    plt.title('SA Learning')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.grid()
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig('figures/nn_sa_learning.png')
    plt.clf()

    # ROC Curve
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    y_probs = model.predicted_probs
    fpr, tpr, _ = roc_curve(test_y, y_probs)
    auc = roc_auc_score(test_y, pred_y)
    plt.plot(fpr, tpr, label='ROC Curve, area={}'.format(auc))
    plt.legend(loc='lower right')
    plt.savefig('figures/nn_sa_roc.png')
    plt.clf()

    print('best classification: ')
    print(classification_report(test_y, pred_y))

def nn_ga(train_x, train_y, test_x, test_y):
    pop_size = [10, 50, 100]
    mutation_prob = [.01, .1, .5]
    # learning_rates = [.01, .1, .5]
    best_auc = 0
    best_pop = 0
    best_mut = 0
    params = []
    times = []
    for ind, i in enumerate(pop_size):
        print(ind/len(pop_size), ' progress')
        for j in mutation_prob:
            start = time.time()
            model = ml.NeuralNetwork(hidden_nodes=[30, 30], activation='relu', algorithm='genetic_alg', max_iters=1000, bias=True, is_classifier=True,
                            learning_rate=0.1, early_stopping=False, pop_size=i, mutation_prob=j, max_attempts=100, random_state=SEED, curve=True)
            model.fit(train_x, train_y)
            # test
            pred_y = model.predict(test_x)
            auc = roc_auc_score(test_y, pred_y)
            times.append(time.time() - start)
            print(auc, ' ', i, ' ', j)
            if auc > best_auc:
                best_pop = i
                best_mut = j
            params.append('pop={}, mut={}'.format(i, j))

    print('times for params: ', list(zip(params, times)))
    print('total time: ', sum(times))


    #Learning Curve
    print('starting plots')
    model = ml.NeuralNetwork(hidden_nodes=[30, 30], activation='relu', algorithm='genetic_alg', max_iters=1000, bias=True, is_classifier=True,
                            learning_rate=0.1, early_stopping=False, pop_size=best_pop, mutation_prob=best_mut, max_attempts=100, random_state=SEED, curve=True)
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, train_x, train_y, n_jobs=N_JOBS,
                       train_sizes=np.linspace(0.1, 1.0, 5), scoring='f1_weighted', verbose=5, random_state=SEED, return_times=True)
    plt.title('GA Learning')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.grid()
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig('figures/nn_ga_learning.png')
    plt.clf()

    # ROC Curve
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    y_probs = model.predicted_probs
    fpr, tpr, _ = roc_curve(test_y, y_probs)
    auc = roc_auc_score(test_y, pred_y)
    plt.plot(fpr, tpr, label='ROC Curve, area={}'.format(auc))
    plt.legend(loc='lower right')
    plt.savefig('figures/nn_ga_roc.png')
    plt.clf()

    print('best classification: ')
    print(classification_report(test_y, pred_y))

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
    elif x == 'n':
        nn_gd(X_train, y_train, X_test, y_test)
