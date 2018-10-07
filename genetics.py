from __future__ import print_function

from pprint import pformat, pprint
from copy import deepcopy
import random
from collections import OrderedDict
from random import randint, randrange
from nets import smallnet
import genes
# import smallnet
# import convnet
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.basicConfig(filename='log.txt', level=logging.DEBUG)


# History of runs: {hash: {chromosome:[..], fitness:3.14, stat:{loss:0.1, val_loss:0.098 ...}}}
Records = dict()


def get_hash(params):
    l = [params['L1']['units'], params['L1']['activation'], 
            params['L2']['units'], params['L2']['activation'],
            params['L3']['activation'], params['opt'], params['lr']]
    return hash(tuple(l))


def create_individual():
    ### smallnet
    params = dict()
    params['L1'] = {'units': genes.get_units(), 'activation': genes.get_activation()}
    params['L2'] = {'units': genes.get_units(), 'activation': genes.get_activation()}
    params['L3'] = {'activation': genes.get_activation()}
    params['opt'] = genes.get_optimizer()
    params['lr'] = genes.get_learning_rate()
    params['hash'] = get_hash(params)
    return params
    # ### lenet
    # params = dict()
    # params['L1'] = {'filters': genes.get_filters(), 'kernel_size': genes.get_kernel_size(), 'activation': genes.get_activation()}
    # params['L2'] = {'filters': genes.get_filters(), 'kernel_size': genes.get_kernel_size(), 'activation': genes.get_activation()}
    # params['L3'] = {'pool_size': genes.get_pool_size()}
    # params['L4'] = {'rate': genes.get_dropout_rate()}
    # params['L5'] = {'units': genes.get_units(), 'activation': genes.get_activation()}
    # params['L6'] = {'rate': genes.get_dropout_rate()}
    # params['L7'] = {'activation': genes.get_activation()}
    # params['opt'] = genes.get_optimizer()
    # params['lr'] = genes.get_learning_rate()
    # return params
    

def update_records(id, indiv, score, stats):
    logging.info('Records updated!')
    Records[id] = {'chromosome': indiv, 'fitness':score, 'stats':stats, 'hash':get_hash(indiv)}
    logging.info("Records[%s] = %s", id, Records[id])


def dummy_fitness(individual):
    # logging.info('dummy_fitness individual: %s', individual)
    score = 1/random.random()
    update_records(len(Records)+1, individual, score, "{'val_loss': 1}")
    return score


def calc_fitness(individual):
    pprint(pformat(individual))
    h = None
    for r in Records.values():
        if r['hash'] == get_hash(individual):
            # score = r.get('score', 0)
            h = r['stats']
    if h is None:
        h = smallnet.run(individual)
    score = 1/h['val_loss'][-1]
    # logging.debug('lenet run completed: %s', h)
    update_records(len(Records)+1, individual, score, h)
    return score


def parent_select(population, fitness, fraction=0.5):
    top_index = int( (1-fraction)*len(population) )
    father, mother = random.sample(population[top_index:], 2)
    return father, mother


def crossover(father, mother):
    # child = [father[key] if random.random() > 0.5 else mother[key] for key in father.keys()]
    mask = [randint(0,1) for _ in range(7)]
    # logging.debug('mask: %s', mask)
    child = [deepcopy(father), deepcopy(mother)]
    # if mask[0] == 1:
    #     child[0]['L1']['units'] = mother['L1']['units']
    #     child[1]['L1']['units'] = father['L1']['units']
    # if mask[1] == 1:
    #     child[0]['L1']['activation'] = mother['L1']['activation']
    #     child[1]['L1']['activation'] = father['L1']['activation']
    # if mask[2] == 1:
    #     child[0]['L2']['units'] = mother['L2']['units']
    #     child[1]['L2']['units'] = father['L2']['units']
    # if mask[3] == 1:
    #     child[0]['L2']['activation'] = mother['L2']['activation']
    #     child[1]['L2']['activation'] = father['L2']['activation']
    # if mask[4] == 1:
    #     child[0]['L3']['activation'] = mother['L3']['activation']
    #     child[1]['L3']['activation'] = father['L3']['activation']
    # if mask[5] == 1:
    #     child[0]['opt'] = mother['opt']
    #     child[1]['opt'] = father['opt']
    # if mask[6] == 1:
    #     child[0]['lr'] = mother['lr']
    #     child[1]['lr'] = father['lr']
    for key in father.keys():
        mask = randint(0,1)
        child[mask][key] = father[key]
        child[1-mask][key] = mother[key]
    # logging.debug('father: %s', father)
    # logging.debug('mother: %s', mother)
    # logging.debug('child[0]: %s', child[0])
    # logging.debug('child[1]: %s', child[1])
    return child



def mutate(indiv, rate):
    if random.random() < rate:
        indiv['L1']['units'] = genes.get_units(exclude=indiv['L1']['units'])
    if random.random() < rate:
        indiv['L1']['activation'] = genes.get_activation(exclude=indiv['L1']['activation'])
    if random.random() < rate:
        indiv['L2']['units'] = genes.get_units(exclude=indiv['L2']['units'])
    if random.random() < rate:
        indiv['L2']['activation'] = genes.get_activation(exclude=indiv['L2']['activation'])
    if random.random() < rate:
        indiv['L3']['activation'] = genes.get_activation(exclude=indiv['L3']['activation'])
    if random.random() < rate:
        indiv['opt'] = genes.get_optimizer(exclude=indiv['opt'])
    if random.random() < rate:
        indiv['lr'] = genes.get_learning_rate()
    return indiv


def main():
    population_size = 20
    tot_generations = 10
    mutate_rate = 0.05
    crossover_rate = 0.5
    population = [create_individual() for _ in range(population_size)]

    for _ in range(tot_generations):
        fitness = [dummy_fitness(p) for p in population]        
        # fitness = [calc_fitness(i) for i in population]
        # sort both fitness & population together
        from operator import itemgetter
        [fitness, population] = [list(x) for x in zip(*sorted(zip(fitness, population), key=itemgetter(0)))]
        # fitness, population = (list(t) for t in zip(*sorted(zip(fitness, population))))
        children = list()
        for _ in range(int(population_size/2)):
            father, mother = parent_select(population, 0.5)
            children.extend(crossover(father, mother))

        for p in range(population_size):
            children[p] = mutate(children[p], mutate_rate)
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
