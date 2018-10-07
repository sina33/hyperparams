from __future__ import print_function

from pprint import pformat, pprint
from copy import deepcopy
import random
from random import randint, randrange
# import lenet
import smallnet
# import convnet
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.basicConfig(filename='log.txt', level=logging.DEBUG)

# Genes
# run(n1=64, n2=64, ac1='relu', ac2='relu', ini1='ones', ini2='zeros', lr=0.1)
Hidden_units = [i for i in range(16, 128)]
Activations = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 
                'hard_sigmoid', 'linear']
Initializers = ['random_normal', 'random_uniform', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'ones', 'zeros']
Learning_rate = [0.1, 0.05, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001]

# History of runs: {hash: {chromosome:[..], fitness:3.14, stat:{loss:0.1, val_loss:0.098 ...}}}
Records = dict()


def create_individual():
    #### smallnet
    n1 = random.choice(Hidden_units)
    n2 = random.choice(Hidden_units)
    ac1 = random.choice(Activations)
    ac2 = random.choice(Activations)
    ini1 = random.choice(Initializers)
    ini2 = random.choice(Initializers)
    lr = random.choice(Learning_rate)
    return [n1, n2, ac1, ac2, ini1, ini2, lr]


def update_records(id, indiv, score, stats):
    logging.info('Records updated!')
    Records[id] = {'chromosome': indiv, 'fitness':score, 'stats':stats}
    logging.info("Records[%s] = %s", id, Records[id])


def dummy_fitness(individual):
    # logging.info('dummy_fitness individual: %s', individual)
    update_records(hash(tuple(individual)), individual, 1, "{'val_loss': 1}")
    return 1


def calc_fitness(individual):
    h = smallnet.run(*individual)
    logging.debug('lenet run completed: %s', h)
    score = 1/h['val_loss'][-1]
    update_records(len(Records)+1, individual, score, h)
    return score


def parent_select(population, fitness, fraction=0.6):
    top_index = int( (1-fraction)*len(population) )
    father, mother = random.sample(population[top_index:], 2)
    return father, mother


def crossover(father, mother):
    crossover_point = random.randint(1,len(father)-1)
    child = [father[i] if i<=crossover_point else mother[i] for i in range(len(father))]
    return child


# run(n1=64, n2=64, ac1='relu', ac2='relu', ini1='ones', ini2='zeros', lr=0.1)
def mutate(indiv, pos):
    # pos = random.choice(range(len(indiv)))
    gene_pool = deepcopy(Hidden_units)
    if pos == 0 or pos == 1:
        pass
    elif pos == 2 or pos == 3:
        gene_pool = deepcopy(Activations)
    elif pos == 4 or pos == 5:
        gene_pool = deepcopy(Initializers)
    elif pos == 6:
        gene_pool = deepcopy(Learning_rate)
    # logging.debug('gene_pool: %s', gene_pool)
    # logging.debug('indiv: %s', indiv)
    # logging.debug('pos: %s', pos)
    # logging.debug('indiv[%s]: %s', pos, indiv[pos])
    gene_pool.remove(indiv[pos])
    # logging.debug('gene_pool: %s', gene_pool)
    new_gene = random.choice(gene_pool)
    indiv[pos] = new_gene
    return indiv


def main():
    population_size = 12
    tot_generations = 10
    mutate_rate = 0.01
    population = [create_individual() for _ in range(population_size)]

    for _ in range(tot_generations):
        fitness = [calc_fitness(i) for i in population]        
        # fitness = [calc_fitness(i) for i in population]
        # sort both fitness & population together
        fitness, population = (list(t) for t in zip(*sorted(zip(fitness, population))))
        children = list()
        for _ in range(population_size):
            father, mother = parent_select(population, 0.5)
            children.append(crossover(father, mother))

        for idx, child in enumerate(children):
            for pos in range(len(child)):
                if random.random() < mutate_rate:
                    children[idx] = mutate(child, pos)

        # for i in range(int(mutate_rate*population_size)):
        #     # logging.debug('children: %s', children)
        #     # logging.debug("children[%s] = %s", i, children[i])
        #     children[i] = mutate(children[i])
        population = children

    # logging.info('Records: %s', pformat(Records))
    c = 1
    s = 0
    for k, v in Records.items():
        logging.info('%s - score: %s, chromosome: %s', c, v['fitness'], v['chromosome'])
        s += v['fitness']
        if c%10 == 0:
            logging.info('mean: %s', s/population_size)
            s = 0
        c += 1
        

if __name__ == '__main__':
    main()
