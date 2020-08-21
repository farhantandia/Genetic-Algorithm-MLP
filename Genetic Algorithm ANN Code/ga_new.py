import numpy 
import random
from numba import jit
import math
import logging
# Converting each solution from matrix to vector.

'''''
##############set logging configurations####################
'''''
logger=logging.getLogger(__name__)

logger.setLevel(level=logging.INFO)

formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s:%(message)s')
file_Handler = logging.FileHandler(filename=__name__+'2.log')
file_Handler.setFormatter(formatter)

logger.addHandler(file_Handler)

'''
##################################################################
'''

class genetic():
    
    learningRate = 0.5

    def __init__(self):

        self.mutateRate = numpy.random.uniform(low=0.1, high=0.5)
        
    def mat_to_vector(self,mat_pop_weights):
        pop_weights_vector = []
        for sol_idx in range(mat_pop_weights.shape[0]):
            curr_vector = []
            for layer_idx in range(mat_pop_weights.shape[1]):
                vector_weights = numpy.reshape(mat_pop_weights[sol_idx, layer_idx], newshape=(mat_pop_weights[sol_idx, layer_idx].size))
                curr_vector.extend(vector_weights)
            pop_weights_vector.append(curr_vector)
        return numpy.array(pop_weights_vector)

    # Converting each solution from vector to matrix.
    def vector_to_mat(self,vector_pop_weights, mat_pop_weights):
        mat_weights = []
        for sol_idx in range(mat_pop_weights.shape[0]):
            start = 0
            end = 0
            for layer_idx in range(mat_pop_weights.shape[1]):
                end = end + mat_pop_weights[sol_idx, layer_idx].size
                curr_vector = vector_pop_weights[sol_idx, start:end]
                mat_layer_weights = numpy.reshape(curr_vector, newshape=(mat_pop_weights[sol_idx, layer_idx].shape))
                mat_weights.append(mat_layer_weights)
                start = end
            #print('Mean and variance of offspring: {} {}'.format(mat_layer_weights[sol_idx,:].mean(), mat_layer_weights[sol_idx,:].std()))
            logger.info('Mean and variance of offspring: {} {}'.format(mat_layer_weights[sol_idx,:].mean(), mat_layer_weights[sol_idx,:].std()))

        return numpy.reshape(mat_weights, newshape=mat_pop_weights.shape)

    def select_mating_pool(self,pop, fitness, num_parents):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((num_parents, pop.shape[1]))
        for parent_num in range(num_parents):
            max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = -99999999999
        return parents

    def crossover(self,parents, offspring_size, crossoverType='onePoint'):
        logger.info('crossoverType:{}'.format(crossoverType))
        if crossoverType=='onePoint':
            offspring = numpy.empty(offspring_size)
            # The point at which crossover takes place between two parents. Usually, it is at the center.
            crossover_point = numpy.uint32(offspring_size[1]/2)
            for k in range(offspring_size[0]):
                # Index of the first parent to mate.
                parent1_idx = k%parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1)%parents.shape[0]
                # The new offspring will have its first half of its genes taken from the first parent.
                offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
                # The new offspring will have its second half of its genes taken from the second parent.
                offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
                logger.info('Mean and Variance of offspring: {} {}'.format(offspring[k,:].mean(), offspring[k,:].std()))

        elif crossoverType=='twoPoint':
            offspring = numpy.empty(offspring_size)
            # The two points at which crossover takes place between two parents.
            crossover_point1 = numpy.uint32(offspring_size[1]*0.25)
            crossover_point2 = numpy.uint32(offspring_size[1]*0.75)
            for k in range(offspring_size[0]):
                # Index of the first parent to mate.
                parent1_idx = k%parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1)%parents.shape[0]
                #take out genes between two points from parent2
                genes_vec2 = parents[parent2_idx, crossover_point1:crossover_point2]
                # The new offspring will have its first part of its genes taken from the first parent,
                #  second part from second parent and final part from first parent again.
                offspring[k, 0:crossover_point1] = parents[parent1_idx, 0:crossover_point1]
                offspring[k,crossover_point1:crossover_point2]= genes_vec2
                # The new offspring will have its second half of its genes taken from the second parent.
                offspring[k, crossover_point2:] = parents[parent1_idx, crossover_point2:]
                logger.info('Mean and Variance of offspring: {} {}'.format(offspring[k,mutation_indices].mean(), offspring[k,mutation_indices].std()))

        return offspring

    def mutation(self,offspring_crossover, mutation_percent, mutationType='adaptive'):
        logger.info('mutationType:{}, mutation_rate:{}'.format(mutationType, mutation_percent))
        num_mutations = numpy.uint32((mutation_percent*offspring_crossover.shape[1])/100)
        mutation_indices = numpy.array(random.sample(range(0, offspring_crossover.shape[1]), num_mutations))
        if mutationType=='uniform':    
            # Mutation changes a single gene in each offspring randomly.
            for idx in range(offspring_crossover.shape[0]):
                # The random value to be added to the gene.
                random_value = numpy.random.uniform(-1.0, 1.0, 1)
                offspring_crossover[idx, mutation_indices] = offspring_crossover[idx, mutation_indices] + random_value
                #print('Mean and Variance of offspring: {} {}'.format(offspring_crossover[idx,mutation_indices].mean(), offspring_crossover[idx,mutation_indices].std()))
                logger.info('Mean and Variance of offspring: {} {}'.format(offspring_crossover[idx,mutation_indices].mean(), offspring_crossover[idx,mutation_indices].std()))

        elif mutationType=='adaptive':
            #implement self-adative mutation
            #print('Mutation Rate: {}'.format(self._mutateRate))
            self.mutateRate = self.mutateRate*math.exp(self.learningRate*numpy.random.uniform(-0.5,0.5))
            logger.info('Mutation Rate new: {}'.format(self.mutateRate))
            for idx in range(offspring_crossover.shape[0]):
                mutation = (self.mutateRate)*(numpy.random.normal(0,0.5))
                offspring_crossover[idx,mutation_indices] = offspring_crossover[idx, mutation_indices] + mutation
                #print('Mean and Variance of offspring: {} {}'.format(offspring_crossover[idx,mutation_indices].mean(), offspring_crossover[idx,mutation_indices].std()))
                logger.info('Mean and Variance of offspring: {} {}'.format(offspring_crossover[idx,mutation_indices].mean(), offspring_crossover[idx,mutation_indices].std()))

        return offspring_crossover
