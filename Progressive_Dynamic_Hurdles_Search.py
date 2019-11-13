import tensorflow as tf
import random
import collections
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

"""
simulation of the progressive dynamic hurdles
count the 1s in bit-string and the one which has most 1s has the high accuracy. 
"""
DIM = 100  # means the number of bit in the bit string that simulate the "models"
STDEV_NOISE = 0.01  # the standard deviation of the simulated training noise
EARLY_SIGNAL_NOISE = 0.005  #
REDUCTION_FACTOR = 100.0  # The factor by which the number of train steps is reduced for earlier observations


class Model(object):
    def __init__(self):
        self.architecture = None
        # this will represent the architecture, in this example is an int that represent
        # the bit-string
        self.observed_accuracy = None
        # the simulated validation accuracy observed for the model
        # during the search. This may be either the accuracy after training for
        # the maximum number of steps or the accuracy after training for 1/100 the
        # maximum number of steps.
        self.true_accuracy = None
        # the accuracy after the maximum train steps


def _sum_bits(arch):
    """Returns the number of 1s in the bit string.

    Args:
      arch: an int representing the bit string.
    """
    total = 0
    for _ in range(DIM):
        total += arch & 1
        # print(total)
        # 按位与运算符：参与运算的两个值,如果两个相应位都为1,则该位的结果为1,否则为0
        arch = (arch >> 1)  # 所有bit 向右一位
    return total


# print(1 << 3)
def get_final_accuracy(arch):
    """Simulates training for the maximum number of steps and then evaluating.

      Args:
        arch: the architecture as an int representing a bit-string.
      """
    accuracy = float(_sum_bits(arch)) / float(DIM)
    accuracy += random.gauss(mu=0.0, sigma=STDEV_NOISE)
    accuracy = 0.0 if accuracy < 0.0 else accuracy
    accuracy = 1.0 if accuracy > 1.0 else accuracy
    return accuracy


def get_early_accuracy(final_accuracy):
    """Simulates training for 1/100 the maximum steps and then evaluating.

    Args:
      final_accuracy: the accuracy of the model if trained for the maximum number
          of steps.
    """
    noise = random.gauss(mu=0,
                         sigma=EARLY_SIGNAL_NOISE)
    observed_accuracy = final_accuracy / REDUCTION_FACTOR + noise
    print(final_accuracy, observed_accuracy, noise)
    observed_accuracy = 0.0 if observed_accuracy < 0.0 else observed_accuracy
    observed_accuracy = 1.0 if observed_accuracy > 1.0 else observed_accuracy
    return observed_accuracy


def random_architecture():
    """Returns a random architecture (bit-string) represented as an int."""
    return random.randint(0, 2 ** DIM - 1)


def mutate_arch(parent_arch):
    """Computes the architecture for a child of the given parent architecture.

    Args:
      parent_arch: an int representing the architecture (bit-string) of the
          parent.

    Returns:
      An int representing the architecture (bit-string) of the child.
    """
    position = random.randint(0, DIM - 1)  # Index of the bit to flip.

    # Flip the bit at position `position` in `child_arch`.
    child_arch = parent_arch ^ (1 << position)
    # 按位异或运算符：当两对应的二进位相异时，结果为1
    # 1 << position: 0000 0001 << 3: 0000 1000

    return child_arch


"""
search algorithms
"""


def plain_evolution(iter, population_size, sample_size, early_observation):
    """
    :param iter: the number of generation
    :param population_size:
    :param sample_size: the sample of parents that needs to be considered to be killed
    :param early_observation: observe the accuracy early or not
    :return:
    """
    # initiation:
    population = collections.deque()  # deque是为了高效实现插入和删除操作的双向列表，适合用于队列和栈
    history = []

    while len(population) < population_size:
        model = Model()
        model.architecture = random_architecture()
        model.true_accuracy = get_final_accuracy(model.architecture)
        # If we are observing early, get the early accuracy that corresponds to the
        # true_accuracy. Else, we are training each model for the maximum number of
        # steps and so the observed_accuracy is the true_accuracy.
        if early_observation:
            model.observed_accuracy = get_early_accuracy(model.true_accuracy)
        else:
            model.observed_accuracy = model.true_accuracy
        history.append(model)
        population.append(model)

    # start evolution:
    while len(history) < iter:
        sample = random.sample(population, sample_size)
        parent = max(sample, key=lambda i: i.observed_accuracy)  # select the best in the sample to be the parent
        child = Model()
        child.architecture = mutate_arch(parent.architecture)
        child.true_accuracy = get_final_accuracy(child.architecture)

        if early_observation:
            child.observed_accuracy = get_early_accuracy(child.true_accuracy)
        else:
            child.observed_accuracy = child.true_accuracy

        min_fitness = float("inf")
        kill_index = population_size
        for s in sample:
            if s.observed_accuracy < min_fitness:
                min_fitness = s.observed_accuracy
                kill_index = population.index(s)
        population[kill_index] = child
        history.append(child)
    return history, population


def pdh_evolution(train_resources, population_size, sample_size):
    """Evolution with PDH.

    Args:
      train_resources: the resources alotted for training. An early obsevation
          costs 1, while a maximum train step observation costs 100.
      population_size: the size of the population.
      sample_size: the size of the sample for both parent selection and killing.
      """
    population = collections.deque()
    history = []  # Not used by the algorithm, only used to report results.
    resources_used = 0  # The number of resource units used.

    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture()
        model.true_accuracy = get_final_accuracy(model.arch)
        # Always initialize with the early observation, since no hurdle has been
        # established.
        model.observed_accuracy = get_early_accuracy(model.true_accuracy)
        population.append(model)
        history.append(model)
        # Since we are only performing an early observation, we are only consuming
        # 1 resource unit.
        resources_used += 1

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    hurdle = None
    while resources_used < train_resources:
        # Sample randomly chosen models from the current population.
        sample = random.sample(population, sample_size)

        # The parent is the best model in the sample, according to the observed
        # accuracy.
        parent = max(sample, key=lambda i: i.observed_accuracy)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch)
        child.true_accuracy = get_final_accuracy(child.arch)
        # Once the hurdle has been established, a model is trained for the maximum
        # amount of train steps if it overcomes the hurdle value. Otherwise, it
        # only trains for the lesser amount of train steps.
        if hurdle:
            child.observed_accuracy = get_early_accuracy(child.true_accuracy)

            # Performing the early observation costs 1 resource unit.
            resources_used += 1
            if child.observed_accuracy > hurdle:
                child.observed_accuracy = child.true_accuracy
                # Now that the model has trained longer, we consume additional
                # resource units.
                resources_used += REDUCTION_FACTOR - 1
        else:
            child.observed_accuracy = get_early_accuracy(child.true_accuracy)
            # Since we are only performing an early observation, we are only consuming
            # 1 resource unit.
            resources_used += 1

        # Choose model to kill.
        sample_indexes = random.sample(range(len(population)), sample_size)
        min_fitness = float("inf")
        kill_index = population_size
        for sample_index in sample_indexes:
            if population[sample_index].observed_accuracy < min_fitness:
                min_fitness = population[sample_index].observed_accuracy
                kill_index = sample_index

        # Replace victim with child.
        population[kill_index] = child

        history.append(child)

        # Create a hurdle, splitting resources such that the number of models
        # trained before and after the hurdle are approximately even. Here, our
        # appoximation is assuming that every model after the hurdle trains for the
        # maximum number of steps.
        if not hurdle and resources_used >= int(train_resources / REDUCTION_FACTOR):
            print("creating new hurdle...")
            hurdle = 0
            for model in population:
                hurdle += model.observed_accuracy
            hurdle /= len(population)

    return history, population


def graph_values(values, title, xlim, ylim):
    plt.figure()
    sns.set_style('white')
    xvalues = range(len(values))
    yvalues = values
    ax = plt.gca()
    dot_size = int(TOTAL_GEN / xlim)
    ax.scatter(
        xvalues, yvalues, marker='.', facecolor=(0.0, 0.0, 0.0),
        edgecolor=(0.0, 0.0, 0.0), linewidth=1, s=dot_size)
    ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=2))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=2))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_title(title, fontsize=20)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.tight_layout()
    ax.tick_params(
        axis='x', which='both', bottom=True, top=False, labelbottom=True,
        labeltop=False, labelsize=14, pad=10)
    ax.tick_params(
        axis='y', which='both', left=True, right=False, labelleft=True,
        labelright=False, labelsize=14, pad=5)
    plt.xlabel('Number of Models Evaluated', labelpad=-16, fontsize=16)
    plt.ylabel('Accuracy', labelpad=-30, fontsize=16)
    plt.xlim(0, xlim + .05)
    plt.ylim(0, ylim + .05)
    sns.despine()
    plt.show()


def graph_history(history):
    observed_accuracies = [i.observed_accuracy for i in history]
    print(max(observed_accuracies))
    true_accuracies = [i.true_accuracy for i in history]
    graph_values(observed_accuracies, "Observed Accuracy",
                 xlim=len(history), ylim=max(observed_accuracies))
    graph_values(true_accuracies, "True Accuracy",
                 xlim=len(history), ylim=max(true_accuracies))


if __name__ == '__main__':
    TOTAL_GEN = 10000  # Total number of resource units.
    POPULATION_SIZE = 100  # The size of the population.
    SAMPLE_SIZE = 10

    history, population = plain_evolution(
        iter=TOTAL_GEN, population_size=POPULATION_SIZE,
        sample_size=SAMPLE_SIZE,
        early_observation=True)

    graph_history(history)
    # history, population = pdh_evolution(train_resources=TOTAL_GEN,
    #                            population_size=POPULATION_SIZE,
    #                            sample_size=SAMPLE_SIZE)
    #
    # graph_history(history)
    #
    # print(len(population), len(history))
