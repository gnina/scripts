#!/usr/bin/env python

'''Train a random forest on model performance from an sql database and then
run a genetic algorithm to propose new, better models to run.
 

'''

import sys, re, MySQLdb, argparse, os, json, subprocess
import pandas as pd
import makemodel
import numpy as np
from MySQLdb.cursors import DictCursor
from outputjson import makejson
from MySQLdb.cursors import DictCursor
from frozendict import frozendict

import sklearn
from sklearn.ensemble import *
from sklearn.preprocessing import *
from sklearn.feature_extraction import *

import deap
from deap import base, creator, gp, tools
from deap import algorithms

from deap import *
import multiprocessing

def getcursor(host,passwd,db):
    '''create a connection and return a cursor;
    doing this guards against dropped connections'''
    conn = MySQLdb.connect (host = host,user = "opter",passwd=passwd,db=db)
    conn.autocommit(True)
    cursor = conn.cursor(DictCursor)
    return cursor
    
def cleanparams(p):
    '''standardize params that do not matter'''
    modeldefaults = makemodel.getdefaults()
    for i in range(1,6):
        if p['conv%d_width'%i] == 0:
            for suffix in ['func', 'init', 'norm', 'size', 'stride', 'width']:
                name = 'conv%d_%s'%(i,suffix)
                p[name] = modeldefaults[name]
        if p['pool%d_size'%i] == 0:
            name = 'pool%d_type'%i
            p[name] = modeldefaults[name]
    if p['fc_pose_hidden'] == 0:
        p['fc_pose_func'] = modeldefaults['fc_pose_func']
        p['fc_pose_hidden2'] = modeldefaults['fc_pose_hidden2']
        p['fc_pose_func2'] = modeldefaults['fc_pose_func2']
        p['fc_pose_init'] = modeldefaults['fc_pose_init']
    elif p['fc_pose_hidden2'] == 0:
        p['fc_pose_hidden2'] = modeldefaults['fc_pose_hidden2']
        p['fc_pose_func2'] = modeldefaults['fc_pose_func2']
        
    if p['fc_affinity_hidden'] == 0:
        p['fc_affinity_func'] = modeldefaults['fc_affinity_func']
        p['fc_affinity_hidden2'] = modeldefaults['fc_affinity_hidden2']
        p['fc_affinity_func2'] = modeldefaults['fc_affinity_func2']
        p['fc_affinity_init'] = modeldefaults['fc_affinity_init']
    elif p['fc_affinity_hidden2'] == 0:
        p['fc_affinity_hidden2'] = modeldefaults['fc_affinity_hidden2']
        p['fc_affinity_func2'] = modeldefaults['fc_affinity_func2']                    
    return p
    
def randParam(param, choices):
    '''randomly select a choice for param'''
    if isinstance(choices, makemodel.Range): #discretize
        choices = np.linspace(choices.min,choices.max, 9)
    return np.asscalar(np.random.choice(choices))

def randomIndividual():
    ret = dict()
    options = makemodel.getoptions()
    for (param,choices) in options.items():
        ret[param] = randParam(param, choices)
    
    return cleanparams(ret)
    
def evaluateIndividual(ind):
    x = dictvec.transform(ind)
    return [rf.predict(x)[0]]
    
def mutateIndividual(ind, indpb=0.05):
    '''for each param, with prob indpb randomly sample another choice'''
    options = makemodel.getoptions()
    for (param,choices) in options.items():
        if np.random.rand() < indpb:
            ind[param] = randParam(param, choices)
    return (ind,)

def crossover(ind1, ind2, indpdb=0.5):
    '''swap choices with probability indpb'''
    options = makemodel.getoptions()
    for (param,choices) in options.items():
        if np.random.rand() < indpdb:
            tmp = ind1[param]
            ind1[param] = ind2[param]
            ind2[param] = tmp
    return (ind1,ind2)   
    
def runGA(pop):
    '''run GA with early stopping if not improving'''    
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    best = 0
    pop = toolbox.clone(pop)
    for i in range(40):
        pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=300, lambda_=300, cxpb=0.5, mutpb=0.2, ngen=25, 
                                       stats=stats, halloffame=hof, verbose=True)
        newmax = log[-1]['max']
        if best == newmax:
            break
        best = newmax
    return pop    
    
def addrows(config,host,db,password):
    '''add rows from fname into database, starting at row start'''
    
    conn = MySQLdb.connect (host = host,user = "opter",passwd=password,db=db)
    cursor = conn.cursor()

    items = list(config.items())
    names = ','.join([str(n) for (n,v) in items])
    values = ','.join(['%s' for (n,v) in items])
    names += ',id'
    values += ',"REQUESTED"'
    
    #do five variations
    for split in range(5):
        seed = np.random.randint(0,100000)
        n = names + ',split,seed'
        v = values + ',%d,%d' % (split,seed) 
        insert = 'INSERT INTO params (%s) VALUES (%s)' % (n,v)
        cursor.execute(insert,[v for (n,v) in items])
            
    conn.commit()
    
                
parser = argparse.ArgumentParser(description='Generate more configurations with random forest and genetic algorithms')
parser.add_argument('--host',type=str,help='Database host',required=True)
parser.add_argument('-p','--password',type=str,help='Database password',required=True)
parser.add_argument('--db',type=str,help='Database name',default='database')
parser.add_argument('--pending_threshold',type=int,default=0,help='Number of pending jobs that triggers an update')
parser.add_argument('-n','--num_configs',type=int,default=1,help='Number of configs to generate - will add a multiple as many jobs') 
args = parser.parse_args()



# first see how many id=REQUESTED jobs there are
cursor = getcursor(args.host,args.password,args.db)
cursor.execute('SELECT COUNT(*) FROM params WHERE id = "REQUESTED"')
rows = cursor.fetchone()
pending = list(rows.values())[0]
#print "Pending jobs:",pending
sys.stdout.write('%d '%pending)
sys.stdout.flush()

#if more than pending_threshold, quit
if pending > args.pending_threshold:
    sys.exit(0)


cursor = getcursor(args.host,args.password,args.db)
cursor.execute('SELECT * FROM params WHERE id != "REQUESTED"')
rows = cursor.fetchall()
data = pd.DataFrame(list(rows))
#make errors zero - appropriate if error is due to parameters
data.loc[data.id == 'ERROR','R'] = 0
data.loc[data.id == 'ERROR','rmse'] = 0
data.loc[data.id == 'ERROR','top'] = 0
data.loc[data.id == 'ERROR','auc'] = 0

data['Rtop'] = data.R*data.top
data = data.dropna('index').apply(pd.to_numeric, errors='ignore')  

#convert data to be useful for sklearn
notparams = ['R','auc','Rtop','id','msg','rmse','seed','serial','time','top','split']
X = data.drop(notparams,axis=1)
y = data.Rtop

dictvec = DictVectorizer()
#standardize meaningless params
Xv = dictvec.fit_transform(list(map(cleanparams,X.to_dict(orient='records'))))

print("\nTraining %d\n"%Xv.shape[0])
#train model
rf = RandomForestRegressor(n_estimators=20)
rf.fit(Xv,y)

#set up GA
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, randomIndividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mutate",mutateIndividual)
toolbox.register("mate",crossover)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluateIndividual)    

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)
    
#setup initial population
initpop = [ creator.Individual(cleanparams(x)) for x in  X.to_dict('records')]

evals = pool.map(toolbox.evaluate, initpop)
top = sorted([l[0] for l in evals],reverse=True)[0]

print("Best in training set: %f"%top)

seen = set(map(frozendict,initpop))
#include some random individuals
randpop = toolbox.population(n=len(initpop))

pop = runGA(initpop+randpop)

#make sure sorted
pop = sorted(pop,key=lambda x: -x.fitness.values[0])
#remove already evaluated configs
pop = [p for p in pop if frozendict(p) not in seen]

print("Best recommended: %f"%pop[0].fitness.values[0])

uniquified = []
for config in pop:
    config = cleanparams(config)
    fr = frozendict(config)
    if fr not in seen:
        seen.add(fr)
        uniquified.append(config)
        
print(len(uniquified),len(pop))        

for config in uniquified[:args.num_configs]:
    addrows(config, args.host,args.db,args.password)
