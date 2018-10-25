from __future__ import print_function

from pprint import pformat, pprint
from copy import deepcopy
import random
from collections import OrderedDict
from random import randint, randrange
import smallnet
import genes
import numpy as np
from operator import itemgetter
import multiprocessing as mp

# import smallnet
# import convnet
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.basicConfig(filename='log.txt', filemode='w', level=logging.DEBUG)


# History of runs: Records[id] = {chromosome:[..], fitness:3.14, stat:{loss:0.1, val_loss:0.098 ...}, hash=1234}
Records = dict()


def is_same(m, n):
    cond = [ n['L1']['units'] == m['L1']['units'] ,
        n['L1']['activation'] == m['L1']['activation'] ,
        n['L2']['units'] == m['L2']['units'] ,
        n['L2']['activation'] == m['L2']['activation'] ,
        n['L3']['activation'] == m['L3']['activation'] ,
        str(n['opt']) == str(m['opt']) ,
        n['lr'] == m['lr'] ]
    return all(cond)
    

def create_individual():
    ### smallnet
    params = dict()
    params['L1'] = {'filters': genes.get_filters(), 'kernel_size': genes.get_kernel_size() , 'activation': genes.get_activation()}
    params['L2'] = {'filters': genes.get_filters(), 'kernel_size': genes.get_kernel_size() , 'activation': genes.get_activation()}
    params['L3'] = {'pool_size': genes.get_pool_size()}
    params['L4'] = {'rate': genes.get_dropout_rate()}
    params['L5'] = {'units': genes.get_units(), 'activation': genes.get_activation()}
    params['L6'] = {'rate': genes.get_dropout_rate()}
    params['L7'] = {'activation': genes.get_activation()}
    params['opt'] = genes.get_optimizer()
    params['lr'] = genes.get_learning_rate()
    # params['hash'] = get_hash(params)
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
    

def update_records(id, indiv, score, err, stats):
    Records[id] = {'fitness': score, 'error': err, 'chromosome': indiv, 'stats':stats}
    # logging.info("Records[%s] = %s", id, Records[id])


def dummy_fitness(individual):
    # logging.info('dummy_fitness individual: %s', individual)
    score = 1/random.random()
    update_records(len(Records)+1, individual, score, 1, "{'val_loss': 1}")
    return score


def calc_fitness(individual):
    h = None
    for r in Records.values():
        if is_same( r['chromosome'], individual):
            h = r['stats']
    if h is None:
        h = smallnet.run(individual)
    score = 1/h['val_loss'][-1]
    l = len(h['val_loss'])
    i = l-1
    while(np.isnan(score) & i>0):
        i -= 1
        score = 1/h['val_loss'][i]
    if np.isnan(score):
        score = 0
    # logging.debug('lenet run completed: %s', h)
    update_records(len(Records)+1, individual, score, 1,  h)
    return score


# get fitness in parallel mode
def get_fitness(individual, q, id, records):
    # pprint(pformat(individual))
    h = lenet.run(individual)
    score = 1/h['val_loss'][-1]
    l = len(h['val_loss'])
    i = l-1
    while(np.isnan(score) & i>0):
        i -= 1
        score = 1/h['val_loss'][i]
    if np.isnan(score):
        score = 0
    else:
        err = round( (1-acc)*100, 2)
        score = round( acc*100, 2)
        # score =  round( 1/loss, 5)
    q.put([id, individual, score, err, h])
    # update_records(len(Records)+1, individual, score, h)


def parent_select(population, fitness, fraction=0.5):
    ### roulette wheel selection
    indices = [i for i in range(len(population))]
    weights = [f for f in fitness]
    [fatherIdx] = random.choices(indices, weights=weights)
    [motherIdx] = random.choices(indices, weights=weights)
    while(motherIdx == fatherIdx):
        [motherIdx] = random.choices(indices, weights=weights)
    # logging.debug('indices: %s', indices)
    # logging.debug('weights: %s', weights)
    # logging.debug('fatherIdx: %s', fatherIdx)
    # logging.debug('motherIdx: %s', motherIdx)
    return population[fatherIdx], population[motherIdx]
    ### truncate selection 
    # top_index = int( (1-fraction)*len(population) )
    # father, mother = random.sample(population[top_index:], 2)
    # return father, mother


def crossover(father, mother):
    # child = [father[key] if random.random() > 0.5 else mother[key] for key in father.keys()]
    mask = [randint(0,1) for _ in range(14)]
    # logging.debug('mask: %s', mask)
    child = [deepcopy(father), deepcopy(mother)]
    if mask[0] == 1:
        child[0]['L1']['filters'] = mother['L1']['filters']
        child[1]['L1']['filters'] = father['L1']['filters']
    if mask[1] == 1:
        child[0]['L1']['kernel_size'] = mother['L1']['kernel_size']
        child[1]['L1']['kernel_size'] = father['L1']['kernel_size']
    if mask[2] == 1:
        child[0]['L1']['activation'] = mother['L1']['activation']
        child[1]['L1']['activation'] = father['L1']['activation']
    if mask[3] == 1:
        child[0]['L2']['filters'] = mother['L2']['filters']
        child[1]['L2']['filters'] = father['L2']['filters']
    if mask[4] == 1:
        child[0]['L2']['kernel_size'] = mother['L2']['kernel_size']
        child[1]['L2']['kernel_size'] = father['L2']['kernel_size']
    if mask[5] == 1:
        child[0]['L2']['activation'] = mother['L2']['activation']
        child[1]['L2']['activation'] = father['L2']['activation']
    if mask[6] == 1:
        child[0]['L3']['pool_size'] = mother['L3']['pool_size']
        child[1]['L3']['pool_size'] = father['L3']['pool_size']
    if mask[7] == 1:
        child[0]['L4']['rate'] = mother['L4']['rate']
        child[1]['L4']['rate'] = father['L4']['rate']
    if mask[8] == 1:
        child[0]['L5']['units'] = mother['L5']['units']
        child[1]['L5']['units'] = father['L5']['units']
    if mask[9] == 1:
        child[0]['L5']['activation'] = mother['L5']['activation']
        child[1]['L5']['activation'] = father['L5']['activation']
    if mask[10] == 1:
        child[0]['L6']['rate'] = mother['L6']['rate']
        child[1]['L6']['rate'] = father['L6']['rate']
    if mask[11] == 1:
        child[0]['L7']['activation'] = mother['L7']['activation']
        child[1]['L7']['activation'] = father['L7']['activation']
    if mask[12] == 1:
        child[0]['opt'] = mother['opt']
        child[1]['opt'] = father['opt']
    if mask[13] == 1:
        child[0]['lr'] = mother['lr']
        child[1]['lr'] = father['lr']
    # for key in father.keys():
    #     mask = randint(0,1)
    #     child[mask][key] = father[key]
    #     child[1-mask][key] = mother[key]

    # logging.debug('father: %s', father)
    # logging.debug('mother: %s', mother)
    # logging.debug('child[0]: %s', child[0])
    # logging.debug('child[1]: %s', child[1])
    return child


def mutate(indiv, rate):
    if random.random() < rate:
        indiv['L1']['filters'] = genes.get_filters(exclude=indiv['L1']['filters'])
    if random.random() < rate:
        indiv['L1']['kernel_size'] = genes.get_kernel_size(exclude=indiv['L1']['kernel_size'])
    if random.random() < rate:
        indiv['L1']['activation'] = genes.get_activation(exclude=indiv['L1']['activation'])
    if random.random() < rate:
        indiv['L2']['filters'] = genes.get_filters(exclude=indiv['L2']['filters'])
    if random.random() < rate:
        indiv['L2']['kernel_size'] = genes.get_kernel_size(exclude=indiv['L2']['kernel_size'])
    if random.random() < rate:
        indiv['L2']['activation'] = genes.get_activation(exclude=indiv['L2']['activation'])
    if random.random() < rate:
        indiv['L3']['pool_size'] = genes.get_pool_size(exclude=indiv['L3']['pool_size'])
    if random.random() < rate:
        indiv['L4']['rate'] = genes.get_dropout_rate()
    if random.random() < rate:
        indiv['L5']['units'] = genes.get_units(exclude=indiv['L5']['units'])
    if random.random() < rate:
        indiv['L5']['activation'] = genes.get_activation(exclude=indiv['L5']['activation'])
    if random.random() < rate:
        indiv['L6']['rate'] = genes.get_dropout_rate()
    if random.random() < rate:
        indiv['L7']['activation'] = genes.get_activation(exclude=indiv['L7']['activation'])
    if random.random() < rate:
        indiv['opt'] = genes.get_optimizer(exclude=indiv['opt'])
    if random.random() < rate:
        indiv['lr'] = genes.get_learning_rate()
    return indiv


def main():
    population_size = 4
    tot_generations = 5
    mutate_rate = 0.15
    crossover_rate = 0.5
    population = [create_individual() for _ in range(population_size)]

    for gen in range(tot_generations):
        queue = mp.Queue()
        processes = [mp.Process(target=get_fitness, args=(p, queue, num+len(Records)+1, Records)) for num, p in enumerate(population)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        fitness = [0] * population_size # list()
        population = [None] * population_size
        while not queue.empty():
            result = queue.get()
            index, chromosome, score, err, hist = result
            update_records(index, chromosome, score, err, hist)
            fitness[index % population_size] = score
            population[index % population_size] = chromosome
        # fitness = [calc_fitness(p) for p in population]        
        # sort both fitness & population together
        [fitness, population] = [list(x) for x in zip(*sorted(zip(fitness, population), key=itemgetter(0)))]
        # convert to descending order
        fitness.reverse()
        population.reverse()

        children = list()
        ### elitism = True
        children.extend(population[:2])
        # for _ in range(int(population_size/2)):
        while(len(children) < population_size):
            father, mother = parent_select(population, fitness)
            children.extend(crossover(father, mother))

        for p in range(population_size):
            children[p] = mutate(children[p], mutate_rate)
        # for i in range(int(mutate_rate*population_size)):
        #     # logging.debug('children: %s', children)
        #     # logging.debug("children[%s] = %s", i, children[i])
        #     children[i] = mutate(children[i])
        population = children[:population_size]


        fitness_sum = 0
        zeroes = 0
        Gen = { k: Records[k] for k in range(gen*population_size+1, (gen+1)*population_size+1) }
        for k, v in Gen.items():
            # logging.info('%s - score: %s, chromosome: %s, stats: %s', k, v['fitness'], v['chromosome'], v['stats'])
            logging.info('%s - score: %s, error: %s, stats: %s', k, v['fitness'], v['error'], v['stats'])
            fitness_sum += v['fitness']
            if v['fitness'] == 0:
                zeroes += 1
        logging.info('  Average Fitness in Generation %s: %s',gen, fitness_sum/(population_size-zeroes))
        logging.info('--'*20)
            

    # # Report
    # c = 1
    # s = 0
    # z = 0
    # for k, v in Records.items():
    #     logging.info('%s - score: %s, chromosome: %s, stats: %s', k, v['fitness'], v['chromosome'], v['stats'])
    #     s += v['fitness']
    #     if v['fitness'] == 0:
    #         z += 1
    #     if c%population_size == 0:
    #         logging.info('mean: %s', s/(population_size-z))
    #         s = 0
    #         z = 0
    #     c += 1

    # ### sort a list of dictionaries in descending order
    # logging.info('='*40)
    # logging.info('sorting Records based on score')
    # logging.info('-'*40)
    # for key, value in sorted(Records.items(), key=lambda x: x[1]['fitness'], reverse=True):
    #     logging.info('Record[%s]: %s', key, value)
    

if __name__ == '__main__':
    main()
