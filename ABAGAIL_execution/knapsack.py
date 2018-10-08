import sys
import os
import time
import csv

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array

# Problem parameters
#=======================
# Random number generator */
random = Random()
# The number of items
NUM_ITEMS = 50
# The number of copies each
COPIES_EACH = 4
# The maximum weight for a single element
MAX_WEIGHT = 50
# The maximum volume for a single element
MAX_VOLUME = 50
# The volume of the knapsack 
KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

# create copies
fill = [COPIES_EACH] * NUM_ITEMS
copies = array('i', fill)

# create weights and volumes
fill = [0] * NUM_ITEMS
weights = array('d', fill)
volumes = array('d', fill)
for i in range(0, NUM_ITEMS):
    weights[i] = random.nextDouble() * MAX_WEIGHT
    volumes[i] = random.nextDouble() * MAX_VOLUME

#print weights
#print volumes
# create range
fill = [COPIES_EACH + 1] * NUM_ITEMS
ranges = array('i', fill)

# Algorithm hyperparameters
#============================
maxiters_rhc= [int(10**(0.25*i)) for i in range(6,26)]
maxiters_sa= [int(10**(0.25*i)) for i in range(6,26)]
maxiters_ga = [int(10**(0.25*i)) for i in range(6,26)]
maxiters_mimic = [10*i for i in range(1,21)]

SA_start_temp = 1E11
SA_temp_decay = 0.999

GA_popsize = 40
GA_toMate = 20
GA_mutationPercent = 0.5
GA_toMutate = int(GA_mutationPercent*GA_toMate)

MIMIC_samples = 500
MIMIC_toKeep = 50
#========================

ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = UniformCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

"""
#=======================
# Random Hiil Climbing
#=======================
print "Starting Random Hill Climbing Seacrh..."
rhc = RandomizedHillClimbing(hcp)
rhc_iters = []
rhc_fitness = []
rhc_time = []
for i in maxiters_rhc:
    fit = FixedIterationTrainer(rhc, i)
    t1=time.time()
    error=fit.train()
    t2=time.time()
    fitness = ef.value(rhc.getOptimal())
    time_ms=round(1000*(t2-t1),2)
    rhc_fitness.append(fitness)
    rhc_time.append(time_ms)
    rhc_iters.append(i)
    #print "RHC: " + str(fitness)
    print "RHC fitness using "+ str(i)+" fixed iterations: " + str(fitness)
    print "Time taken for RHC using fixed iterations: "+str(time_ms)+" milliseconds"

print "Finished Random Hill Climbing Seacrh."
print "="*100
"""

"""
#=======================
# Simulated Annealing
#=======================
print "Starting Simulated Annealing Seacrh..."
sa = SimulatedAnnealing(SA_start_temp, SA_temp_decay, hcp)
sa_iters = []
sa_fitness = []
sa_time = []

for i in maxiters_sa:
    fit = FixedIterationTrainer(sa, i)
    t1=time.time()
    fit.train()
    t2=time.time()
    fitness = ef.value(sa.getOptimal())
    time_ms=round(1000*(t2-t1),2)
    sa_fitness.append(fitness)
    sa_time.append(time_ms)
    sa_iters.append(i)
    print "SA fitness using "+ str(i)+" fixed iterations: " + str(fitness)
    print "Time taken for SA using fixed iterations: "+str(time_ms)+" milliseconds"

print "Finished Simulated Annealing Seacrh."
print "="*100
"""

"""
#=======================
# Genetic Algorithm
#=======================
print "Starting Genetic Algorithm Seacrh..."
ga = StandardGeneticAlgorithm(GA_popsize, GA_toMate, GA_toMutate, gap)
ga_iters = []
ga_fitness = []
ga_time = []

for i in maxiters_ga:
    fit = FixedIterationTrainer(ga, i)
    t1=time.time()
    fit.train()
    t2=time.time()
    fitness = ef.value(ga.getOptimal())
    time_ms=round(1000*(t2-t1),2)
    ga_fitness.append(fitness)
    ga_time.append(time_ms)
    ga_iters.append(i)
    print "GA fitness using "+ str(i)+" fixed iterations: " + str(fitness)
    print "Time taken for GA using fixed iterations: "+str(time_ms)+" milliseconds"

print "Finished Genetic Algorithm Seacrh."
print "="*100
"""

#"""
#=======================
# MIMIC
#=======================
print "Starting MIMIC Seacrh..."
mimic = MIMIC(MIMIC_samples, MIMIC_toKeep, pop)
mimic_iters = []
mimic_fitness = []
mimic_time = []

for i in maxiters_mimic:
    fit = FixedIterationTrainer(mimic, i)
    t1=time.time()
    fit.train()
    t2=time.time()
    fitness = ef.value(mimic.getOptimal())
    time_ms=round(1000*(t2-t1),2)
    mimic_fitness.append(fitness)
    mimic_time.append(time_ms)
    mimic_iters.append(i)
    print "MIMIC fitness using "+ str(i)+" fixed iterations: " + str(fitness)
    print "Time taken for MIMIC using fixed iterations: "+str(time_ms)+" milliseconds"
    
print "Finished MIMIC Seacrh."
print "="*100
#"""

"""
# Writing RHC performance to a CSV
spamWriter = csv.writer(open('knapsack_rhc.csv', 'w'), delimiter=' ',quotechar='|')
spamWriter.writerow(rhc_iters)
spamWriter.writerow(rhc_fitness)
spamWriter.writerow(rhc_time)

# Writing SA performance to a CSV
spamWriter = csv.writer(open('knapsack_sa.csv', 'w'), delimiter=' ',quotechar='|')
spamWriter.writerow(sa_iters)
spamWriter.writerow(sa_fitness)
spamWriter.writerow(sa_time)

# Writing GA performance to a CSV
spamWriter = csv.writer(open('knapsack_ga.csv', 'w'), delimiter=' ',quotechar='|')
spamWriter.writerow(ga_iters)
spamWriter.writerow(ga_fitness)
spamWriter.writerow(ga_time)
"""
# Writing MIMIC performance to a CSV
spamWriter = csv.writer(open('knapsack_mimic.csv', 'w'), delimiter=' ',quotechar='|')
spamWriter.writerow(mimic_iters)
spamWriter.writerow(mimic_fitness)
spamWriter.writerow(mimic_time)
#"""
