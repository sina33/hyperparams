from __future__ import print_function

from pprint import pformat, pprint
from copy import deepcopy
import random
from random import randint, randrange
import lenet
import genes
# import smallnet
# import convnet
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.basicConfig(filename='log.txt', level=logging.DEBUG)


# History of runs: {hash: {chromosome:[..], fitness:3.14, stat:{loss:0.1, val_loss:0.098 ...}}}
Records = dict()


def create_individual():
    ### lenet
    params = dict()
    params['L1'] = {'filters': genes.get_filters(), 'kernel_size': genes.get_kernel_size(), 'activation': genes.get_activation()}
    params['L2'] = {'filters': genes.get_filters(), 'kernel_size': genes.get_kernel_size(), 'activation': genes.get_activation()}
    params['L3'] = {'pool_size': genes.get_pool_size()}
    params['L4'] = {'rate': genes.get_dropout_rate()}
    params['L5'] = {'units': genes.get_units(), 'activation': genes.get_activation()}
    params['L6'] = {'rate': genes.get_dropout_rate()}
    params['L7'] = {'activation': genes.get_activation()}
    params['opt'] = genes.get_optimizer()
    params['lr'] = genes.get_learning_rate()
    return params
    

def update_records(id, indiv, score, stats):
    logging.info('Records updated!')
    Records[id] = {'chromosome': indiv, 'fitness':score, 'stats':stats}
    logging.info("Records[%s] = %s", id, Records[id])


def dummy_fitness(individual):
    # logging.info('dummy_fitness individual: %s', individual)
    update_records(hash(tuple(individual)), individual, 1, "{'val_loss': 1}")
    return 1



def calc_fitness(individual):
    pprint(pformat(individual))
    h = None
    for r in Records.values():
        if hash(tuple(r['chromosome'])) == hash(tuple(individual)):
            h = r['stats']
    if h is None:
        h = lenet.run(individual)
    logging.debug('lenet run completed: %s', h)
    score = 1/h['val_loss'][-1]
    update_records(len(Records)+1, individual, score, h)
    return score


def parent_select(population, fitness, fraction=0.5):
    top_index = int( (1-fraction)*len(population) )
    father, mother = random.sample(population[top_index:], 2)
    return father, mother


def crossover(father, mother):
    # child = [father[key] if random.random() > 0.5 else mother[key] for key in father.keys()]
    child = [father, mother]
    for key in father.keys():
        mask = randint(0,1)
        child[mask][key] = father[key]
        child[1-mask][key] = mother[key]
    return child



def mutate(indiv):
    return indiv


def main():
    population_size = 12
    tot_generations = 10
    mutate_rate = 0.1
    crossover_rate = 0.5
    population = [create_individual() for _ in range(population_size)]

    for _ in range(tot_generations):
        fitness = [calc_fitness(i) for i in population]        
        # fitness = [calc_fitness(i) for i in population]
        # sort both fitness & population together
        fitness, population = (list(t) for t in zip(*sorted(zip(fitness, population))))
        children = list()
        for _ in range(population_size):
            father, mother = parent_select(population, 0.5)
            children.extend(crossover(father, mother))

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
