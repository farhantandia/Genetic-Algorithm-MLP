import logging
import pickle
import time

import matplotlib.pyplot as plt
import numpy
from ann_new import ANN
from ga_new import genetic
from sklearn.model_selection import train_test_split

"""''
##############set logging configurations####################
""" ""
logger = logging.getLogger(__name__)

logger.setLevel(level=logging.INFO)

formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(name)s:%(message)s")
file_Handler = logging.FileHandler(filename="GA-ANN.log")
file_Handler.setFormatter(formatter)

logger.addHandler(file_Handler)

"""
##################################################################
"""
# set numpy random number generator seed

numpy.random.seed(1234)


class dataset:
    def __init__(self):

        self.data_inputs = 0
        self.data_outputs = 0

        with open("data/dataset_features_6.pkl", "rb") as f:
            self.data_inputs = pickle.load(f)
        # self.features_STDs = numpy.std(a=self.data_inputs2, axis=0)
        # self.data_inputs = self.data_inputs2[:, self.features_STDs>50]

        with open("data/outputs_6.pkl", "rb") as f:
            self.data_outputs = pickle.load(f)
        # self.data_inputs = self.data_inputs[:, self.features_STDs > 50]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data_inputs, self.data_outputs, test_size=0.25
        )


class genetic_network:

    """
    Genetic algorithm parameters:
        Mating Pool Size (Number of Parents)
        Population Size
        Number of Generations
        Mutation Percent
    """

    def __init__(
        self,
        dataset,
        sol_per_pop=50,
        num_parents_mating=25,
        num_generations=1,
        mutation_rate=30,
        crossoverType="onePoint",
        mutationType="adaptive",
    ):

        self.sol_per_pop = sol_per_pop
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.initial_pop_weights = []
        self.fitness = 0
        self.accuracies = 0
        self.crossoverType = crossoverType
        self.mutationType = mutationType
        self.predictions = 0
        self.data = dataset
        self.input_shape = self.data.X_train.shape[1]
        self.HL1_neurons = 128
        self.HL2_neurons = 256
        self.HL3_neurons = 256
        self.output_neurons = 6

    def population(self, data):

        # Creating the initial population.
        for curr_sol in numpy.arange(0, self.sol_per_pop):

            input_HL1_weights = numpy.random.uniform(
                low=-0.1, high=0.1, size=(self.input_shape, self.HL1_neurons)
            )

            HL1_HL2_weights = numpy.random.uniform(
                low=-0.1, high=0.1, size=(self.HL1_neurons, self.HL2_neurons)
            )

            HL2_HL3_weights = numpy.random.uniform(
                low=-0.1, high=0.1, size=(self.HL2_neurons, self.HL3_neurons)
            )

            HL3_output_weights = numpy.random.uniform(
                low=-0.1, high=0.1, size=(self.HL3_neurons, self.output_neurons)
            )

            self.initial_pop_weights.append(
                numpy.array(
                    [
                        input_HL1_weights,
                        HL1_HL2_weights,
                        HL2_HL3_weights,
                        HL3_output_weights,
                    ]
                )
            )
        logger.info(
            "solution_per_pop: {}, num_parents_mating: {}, num_generations: {}, mutation_rate: {}, mutationType: {}, crossType: {}".format(
                self.sol_per_pop,
                self.num_parents_mating,
                self.num_generations,
                self.mutation_rate,
                self.mutationType,
                self.crossoverType,
            )
        )
        logger.info(
            "ANN Layers: {}{}{}".format(
                input_HL1_weights.shape, HL1_HL2_weights.shape, HL3_output_weights.shape
            )
        )

    def evolve(self, data):

        # create instance of genetic class
        ga = genetic()
        # create dataset instance
        data = dataset()
        pop_weights_mat = numpy.array(self.initial_pop_weights)
        pop_weights_vector = ga.mat_to_vector(pop_weights_mat)

        best_outputs = []
        self.accuracies = numpy.empty(shape=(self.num_generations))
        start_time = time.time()

        for generation in range(self.num_generations):

            # converting the solutions from being vectors to matrices.
            pop_weights_mat = ga.vector_to_mat(pop_weights_vector, pop_weights_mat)

            # Measuring the fitness of each chromosome in the population.
            self.fitness = ANN.fitness(
                pop_weights_mat, data.X_train, data.y_train, activation="relu"
            )
            self.accuracies[generation] = self.fitness[0]

            logger.info(
                "Generation: {} Max Fitness: {}".format(
                    generation, numpy.max(self.fitness)
                )
            )
            print(
                "Generation: {} Max Fitness: {}".format(
                    generation, numpy.max(self.fitness)
                )
            )

            # Selecting the best parents in the population for mating.
            parents = ga.select_mating_pool(
                pop_weights_vector, self.fitness.copy(), self.num_parents_mating
            )

            # Generating next generation using crossover.
            offspring_crossover = ga.crossover(
                parents,
                offspring_size=(
                    pop_weights_vector.shape[0] - parents.shape[0],
                    pop_weights_vector.shape[1],
                ),
                crossoverType=self.crossoverType,
            )

            # Adding some variations to the offsrping using mutation.
            offspring_mutation = ga.mutation(
                offspring_crossover,
                mutation_percent=self.mutation_rate,
                mutationType=self.mutationType,
            )

            # Creating the new population based on the parents and offspring.
            pop_weights_vector[0 : parents.shape[0], :] = parents
            pop_weights_vector[parents.shape[0] :, :] = offspring_mutation

        pop_weights_mat = ga.vector_to_mat(pop_weights_vector, pop_weights_mat)
        best_weights = pop_weights_mat[0, :]
        acc, self.predictions = ANN.predict_outputs(
            best_weights, data.X_test, data.y_test, activation="relu"
        )
        print("Accuracy of the best solution is : ", acc)

        elapsed_time = time.time() - start_time
        if elapsed_time < 100:
            print("Elapsed time for processing in second:", elapsed_time)
            logger.info(
                "Elapsed time for processing in second: {}".format(elapsed_time)
            )

        else:
            elapsed_time = elapsed_time / 60
            print("Elapsed time for processing in minute: ", elapsed_time)
            logger.info(
                "Elapsed time for processing in minute: {}".format(elapsed_time)
            )

        with open(
            f"weights_{self.num_generations}_iterations_{self.mutation_rate}_mutation.pkl",
            "wb",
        ) as f:
            pickle.dump(pop_weights_mat, f)

    def plot(self):

        plt.plot(self.accuracies)
        plt.xlabel("Generation")
        plt.ylabel("Fitness(Accuracy")
        plt.xticks(numpy.arange(0, self.num_generations + 1, 100))
        plt.xlim(0, self.num_generations)
        plt.yticks(numpy.arange(0, 101, 10))
        plt.savefig("GA_results.png")
        plt.show()


def main():

    data = dataset()
    gn = genetic_network(dataset=data)
    gn.population(data=data)
    gn.evolve(data=data)
    # gn.plot()

    # Calculating some statistics
    num_correct = sum(numpy.array(gn.predictions) == numpy.array(data.y_test))
    num_wrong = data.y_test.size - num_correct
    accuracy = 100 * (num_correct / data.y_test.size)

    print()
    print(
        "sol per pop = {} , n parent mating = {}, n generation = {}, mutation rate = {}".format(
            gn.sol_per_pop, gn.num_parents_mating, gn.num_generations, gn.mutation_rate
        )
    )
    logger.info(
        "sol per pop = {} , n parent mating = {}, n generation = {}, mutation rate = {}".format(
            gn.sol_per_pop, gn.num_parents_mating, gn.num_generations, gn.mutation_rate
        )
    )
    print()
    print("Number of correct classifications: {}.".format(num_correct))
    logger.info("Number of correct classifications: {}.".format(num_correct))

    print("Number of wrong classifications : {}.".format(num_wrong))
    logger.info("Number of wrong classifications : {}.".format(num_wrong))

    print("Classification accuracy : {}.".format(accuracy))
    logger.info("Classification accuracy : {}.".format(accuracy))


if __name__ == "__main__":
    main()
