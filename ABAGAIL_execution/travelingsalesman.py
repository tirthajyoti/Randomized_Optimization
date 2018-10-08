# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes 
# to a file and plot them in your favorite tool.
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
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
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
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array

# Problem parameters
#=======================
# set N value.  This is the number of points
N = 50
random = Random()

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()

# Algorithm hyperparameters
#============================
maxiters_rhc= [int(10**(0.25*i)) for i in range(6,26)]
maxiters_sa= [int(10**(0.25*i)) for i in range(6,26)]
maxiters_ga = [int(10**(0.25*i)) for i in range(4,21)]
maxiters_mimic = [10*i for i in range(1,21)]

SA_start_temp = 1E11
SA_temp_decay = 0.999

GA_popsize = 50
GA_toMate = 25
GA_mutationPercent = 0.5
GA_toMutate = int(GA_mutationPercent*GA_toMate)

MIMIC_samples = 400
MIMIC_toKeep = 40
#========================
ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscretePermutationDistribution(N)
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

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
    print "RHC inverse of distance using "+ str(i)+" fixed iterations: " + str(fitness)
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
    print "SA inverse of distance using "+ str(i)+" fixed iterations: " + str(fitness)
    print "Time taken for SA using fixed iterations: "+str(time_ms)+" milliseconds"

print "Finished Simulated Annealing Seacrh."
print "="*100
"""

#"""
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
    print "GA inverse of distance using "+ str(i)+" fixed iterations: " + str(fitness)
    print "Time taken for GA using fixed iterations: "+str(time_ms)+" milliseconds"

print "Finished Genetic Algorithm Seacrh."
print "="*100
#"""

"""
#=======================
# MIMIC
#=======================
# for mimic we use a sort encoding
ef = TravelingSalesmanSortEvaluationFunction(points);
fill = [N] * N
ranges = array('i', fill)
odd = DiscreteUniformDistribution(ranges);
df = DiscreteDependencyTree(.1, ranges); 
pop = GenericProbabilisticOptimizationProblem(ef, odd, df);

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
    print "MIMIC inverse of distance using "+ str(i)+" fixed iterations: " + str(fitness)
    print "Time taken for MIMIC using fixed iterations: "+str(time_ms)+" milliseconds"
    
print "Finished MIMIC Seacrh."
print "="*100
"""

"""
# Writing RHC performance to a CSV
spamWriter = csv.writer(open('tsp_rhc.csv', 'w'), delimiter=' ',quotechar='|')
spamWriter.writerow(rhc_iters)
spamWriter.writerow(rhc_fitness)
spamWriter.writerow(rhc_time)
"""

"""
# Writing SA performance to a CSV
spamWriter = csv.writer(open('tsp_sa.csv', 'w'), delimiter=' ',quotechar='|')
spamWriter.writerow(sa_iters)
spamWriter.writerow(sa_fitness)
spamWriter.writerow(sa_time)
"""

#"""
# Writing GA performance to a CSV
spamWriter = csv.writer(open('tsp_ga.csv', 'w'), delimiter=' ',quotechar='|')
spamWriter.writerow(ga_iters)
spamWriter.writerow(ga_fitness)
spamWriter.writerow(ga_time)
#"""

"""
# Writing MIMIC performance to a CSV
spamWriter = csv.writer(open('tsp_mimic.csv', 'w'), delimiter=' ',quotechar='|')
spamWriter.writerow(mimic_iters)
spamWriter.writerow(mimic_fitness)
spamWriter.writerow(mimic_time)
"""

