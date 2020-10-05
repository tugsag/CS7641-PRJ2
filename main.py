import mlrose_hiive as ml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

SEED = 903454028
np.random.seed(SEED)

def plot(x, y, title, xlabel, ylabel, path=None, ticks=None, labels=None):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if ticks is not None:
        plt.xticks(ticks=ticks, labels=labels, rotation=90)
    plt.tight_layout()
    plt.savefig('figures/'+ path)
    plt.clf()

def hill_climb(fit, id=None):
    restarts = [10, 25, 50, 100, 200]
    if id == 'f':
        name = 'flipflop'
        problem = ml.DiscreteOpt(length=300, fitness_fn=fit)
    elif id == 'k':
        name = 'knap'
        problem = ml.DiscreteOpt(length=100, fitness_fn=fit, max_val=2)
    else:
        name = 'queens'
        problem = ml.DiscreteOpt(length=32, fitness_fn=fit, max_val=32)
    states = []
    fitnesses = []
    curves = []
    times = []
    for i in restarts:
        start = time.time()
        state, fitness, curve = ml.random_hill_climb(problem=problem, max_attempts=100, max_iters=7500, restarts=i, curve=True, random_state=SEED)
        states.append(state)
        fitnesses.append(fitness)
        curves.append(curve)
        end = time.time()
        times.append(end - start)
    fitnesses = np.array(fitnesses)
    curves = np.array(curves)
    means = [np.mean(i) for i in curves]
    index = np.argmax(means)
    best_curve = curves[index]
    print('max fitness is ', fitnesses.max())
    print('max fitness params ', restarts[np.argmax(fitnesses)])
    print('mean of best curve ', means[index])
    print('best param: ', restarts[index])
    plot(range(len(best_curve)), best_curve, 'Best Curve - RHC', 'Iterations', 'Fitness', path='{}_rhc_best_curve.png'.format(name))
    plot(restarts, fitnesses, 'Restarts vs Fitness - RHC', 'Restarts', 'Fitness', path='{}_rhc_fitness.png'.format(name))
    plot(restarts, times, 'Restarts vs Time - RHC', 'Restarts', 'Time', path='{}_rhc_time.png'.format(name))

def annealing(fit, id=None):
    temps = [0.1, 0.5, 1, 10, 25, 50, 100]
    if id == 'f':
        name = 'flipflop'
        problem = ml.DiscreteOpt(length=300, fitness_fn=fit)
    elif id == 'k':
        name = 'knap'
        problem = ml.DiscreteOpt(length=100, fitness_fn=fit, max_val=2)
    else:
        name = 'queens'
        problem = ml.DiscreteOpt(length=32, fitness_fn=fit, max_val=32)
    # Exp Decay
    fitnesses_e = []
    curves_e = []
    times_e = []
    for i in temps:
        start = time.time()
        schedule = ml.ExpDecay(init_temp=i)
        state, fitness, curve = ml.simulated_annealing(problem=problem, schedule=schedule, max_attempts=100, max_iters=7500, curve=True, random_state=SEED)
        fitnesses_e.append(fitness)
        curves_e.append(curve)
        end = time.time()
        times_e.append(end - start)
    fitnesses_e = np.array(fitnesses_e)
    curves_e = np.array(curves_e)
    means_e = [np.mean(i) for i in curves_e]
    index = np.argmax(means_e)
    best_curve = curves_e[index]
    print('max fitness is ', fitnesses_e.max())
    print('max fitness params ', temps[np.argmax(fitnesses_e)])
    print('mean of best curve ', means_e[index])
    print('best param: ', temps[index])
    plot(range(len(best_curve)), best_curve, 'Best Curve - Exp SA', 'Iterations', 'Fitness', path='{}_sa_exp_best_curve.png'.format(name))
    plot(temps, fitnesses_e, 'Init Temps vs Fitness - Exp SA', 'Temps', 'Fitness', path='{}_sa_exp_fitness.png'.format(name))
    plot(temps, times_e, 'Init Temps vs Time - Exp SA', 'Temps', 'Time', path='{}_sa_exp_time.png'.format(name))

    # Geo Decay
    fitnesses_g = []
    curves_g = []
    times_g = []
    for i in temps:
        start = time.time()
        schedule = ml.GeomDecay(init_temp=i)
        state, fitness, curve = ml.simulated_annealing(problem=problem, schedule=schedule, max_attempts=100, max_iters=7500, curve=True, random_state=SEED)
        fitnesses_g.append(fitness)
        curves_g.append(curve)
        end = time.time()
        times_g.append(end - start)
    fitnesses_g = np.array(fitnesses_g)
    curves_g = np.array(curves_g)
    means_g = [np.mean(i) for i in curves_g]
    index = np.argmax(means_g)
    best_curve = curves_g[index]
    print('max fitness is ', fitnesses_g.max())
    print('max fitness params ', temps[np.argmax(fitnesses_g)])
    print('mean of best curve ', means_g[index])
    print('best param: ', temps[index])
    plot(range(len(best_curve)), best_curve, 'Best Curve - Geo SA', 'Iterations', 'Fitness', path='{}_sa_geo_best_curve.png'.format(name))
    plot(temps, fitnesses_g, 'Init Temps vs Fitness - Geo SA', 'Temps', 'Fitness', path='{}_sa_geo_fitness.png'.format(name))
    plot(temps, times_g, 'Init Temps vs Time - Geo SA', 'Temps', 'Time', path='{}_sa_geo_time.png'.format(name))

def genetic(fit, id=None):
    pop_size = [50, 100, 200, 350, 500]
    mutation_prob = [.05, .1, .2, .5]
    if id == 'f':
        name = 'flipflop'
        problem = ml.DiscreteOpt(length=300, fitness_fn=fit)
    elif id == 'k':
        name = 'knap'
        problem = ml.DiscreteOpt(length=100, fitness_fn=fit, max_val=2)
    else:
        name = 'queens'
        problem = ml.DiscreteOpt(length=32, fitness_fn=fit, max_val=32)
    fitnesses = []
    curves = []
    times = []
    params = []
    for i in pop_size:
        for j in mutation_prob:
            start = time.time()
            state, fitness, curve = ml.genetic_alg(problem=problem, pop_size=i, mutation_prob=j, max_attempts=100, max_iters=7500, curve=True, random_state=SEED)
            fitnesses.append(fitness)
            curves.append(curve)
            params.append('pop_size={}, mutation={}'.format(i, j))
            end = time.time()
            times.append(end - start)
    fitnesses = np.array(fitnesses)
    curves = np.array(curves)
    means = [np.mean(i) for i in curves]
    index = np.argmax(means)
    best_curve = curves[index]
    print('max fitness is ', fitnesses.max())
    print('max fitness params ', params[np.argmax(fitnesses)])
    print('mean of best curve ', means[index])
    print('best param: ', params[index])
    plot(range(len(best_curve)), best_curve, 'Best Curve - GA', 'Iterations', 'Fitness', path='{}_ga_best_curve.png'.format(name))
    plot(params, fitnesses, 'Pop Size & Mutation Prob vs Fitness - GA', 'Pop Size & Mutation Prob', 'Fitness', path='{}_ga_fitness.png'.format(name), ticks=range(len(params)), labels=params)
    plot(params, times, 'Pop Size & Mutation Prob vs Time - GA', 'Pop Size & Mutation Prob', 'Time', path='{}_ga_time.png'.format(name), ticks=range(len(params)), labels=params)

def mimic(fit, id=None):
    pop_size = [100, 200, 500]
    keep_pct = [.05, .1, .5]
    if id == 'f':
        name = 'flipflop'
        problem = ml.DiscreteOpt(length=300, fitness_fn=fit)
    elif id == 'k':
        name = 'knap'
        problem = ml.DiscreteOpt(length=100, fitness_fn=fit, max_val=2)
    else:
        name = 'queens'
        problem = ml.DiscreteOpt(length=32, fitness_fn=fit, max_val=32)
    fitnesses = []
    curves = []
    times = []
    params = []
    for ind, i in enumerate(pop_size):
        print(ind/len(pop_size), ' progress')
        for j in keep_pct:
            print(j)
            start = time.time()
            state, fitness, curve = ml.mimic(problem=problem, pop_size=i, keep_pct=j, max_attempts=100, max_iters=7500, curve=True, random_state=SEED)
            fitnesses.append(fitness)
            curves.append(curve)
            params.append('pop_size={}, keep_pct={}'.format(i, j))
            end = time.time()
            times.append(end - start)
    fitnesses = np.array(fitnesses)
    curves = np.array(curves)
    means = [np.mean(i) for i in curves]
    index = np.argmax(means)
    best_curve = curves[index]
    print('max fitness is ', fitnesses.max())
    print('max fitness params ', params[np.argmax(fitnesses)])
    print('mean of best curve ', means[index])
    print('best param: ', params[index])
    plot(range(len(best_curve)), best_curve, 'Best Curve - Mimic', 'Iterations', 'Fitness', path='{}_mimic_best_curve.png'.format(name))
    plot(params, fitnesses, 'Pop Size & Keep_pct vs Fitness - Mimic', 'Pop Size & Keep_pct', 'Fitness', path='{}_mimic_fitness.png'.format(name), ticks=range(len(params)), labels=params)
    plot(params, times, 'Pop Size & Keep_pct vs Time - Mimic', 'Pop Size & Keep_pct', 'Time', path='{}_mimic_time.png'.format(name), ticks=range(len(params)), labels=params)

# helper
def max_queens(state):
    fitness = 0
    for i in range(len(state) - 1):
        for j in range(i+1, len(state)):
            if (state[j] != state[i]) and (state[j] != state[i] + (j-i)) and (state[j] != state[i] - (j-i)):
                fitness += 1
    return fitness


if __name__ == '__main__':
    x = input('''Choose problem:
                    FlipFlop: f,
                    Knapsack: k,
                    Queens: q: ''')
    y = input('''Choose algorithm:
                    Hill climb: R,
                    Annealing: A,
                    Genetic: G,
                    MIMIC: M: ''')
    if x == 'f':
        fit = ml.FlipFlop()
        if y == 'R':
            hill_climb(fit, id=x)
        elif y == 'A':
            annealing(fit, id=x)
        elif y == 'G':
            genetic(fit, id=x)
        elif y == 'M':
            mimic(fit, id=x)
    elif x == 'k':
        weights = np.random.randint(10, 40, 100)
        values = np.random.randint(20, 30, 100)
        fit = ml.Knapsack(weights=weights, values=values, max_weight_pct=.6)
        if y == 'R':
            hill_climb(fit, id=x)
        elif y == 'A':
            annealing(fit, id=x)
        elif y == 'G':
            genetic(fit, id=x)
        elif y == 'M':
            mimic(fit, id=x)
    elif x == 'q':
        fit = ml.CustomFitness(max_queens)
        if y == 'R':
            hill_climb(fit, id=x)
        elif y == 'A':
            annealing(fit, id=x)
        elif y == 'G':
            genetic(fit, id=x)
        elif y == 'M':
            mimic(fit, id=x)
