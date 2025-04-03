#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Evolve a better keyboard.
This assignment is mostly open-ended,
with a couple restrictions:

# DO NOT MODIFY >>>>
Do not edit the sections between these marks below.
# <<<< DO NOT MODIFY
"""

# %%
import random
from typing import TypedDict
import math
import json
import datetime
import numpy as np

# DO NOT MODIFY >>>>
# First, what should our representation look like?
# Is there any modularity in adjacency?
# What mechanisms capitalize on such modular patterns?
# ./corpus/2_count.py specificies this same structure:
# Positions =  01234   56789   01234
LEFT_DVORAK = "',.PY" "AOEUI" ";QJKX"
LEFT_QWERTY = "QWERT" "ASDFG" "ZXCVB"
LEFT_COLEMK = "QWFPG" "ARSTD" "ZXCVB"
LEFT_WORKMN = "QDRWB" "ASHTG" "ZXMCV"

LEFT_DISTAN = "22222" "11112" "22222"
LEFT_ERGONO = "11112" "11112" "22323"
LEFT_EDGE_B = "12345" "12345" "12345"

# 8 non-thumb fingers are used for touch typing:
LEFT_FINGID = "01233" "01233" "01233"

# Positions     56   7890123   456789   01234
RIGHT_DVORAK = "[]" "FGCRL/=" "DHTNS-" "BMWVZ"
RIGHT_QWERTY = "-=" "YUIOP[]" "HJKL;'" "NM,./"
RIGHT_COLEMK = "-=" "JLUY;[]" "HNEIO'" "KM,./"
RIGHT_WOKRMN = "-=" "JFUP;[]" "YNEOI'" "KL,./"

RIGHT_DISTAN = "34" "2222223" "211112" "22222"
RIGHT_ERGONO = "33" "3111134" "211112" "21222"
RIGHT_EDGE_B = "21" "7654321" "654321" "54321"

# Non-thumb fingers are numbered [0,...,7]
RIGHT_FINGID = "77" "4456777" "445677" "44567"

DVORAK = LEFT_DVORAK + RIGHT_DVORAK
QWERTY = LEFT_QWERTY + RIGHT_QWERTY
COLEMAK = LEFT_COLEMK + RIGHT_COLEMK
WORKMAN = LEFT_WORKMN + RIGHT_WOKRMN

DISTANCE = LEFT_DISTAN + RIGHT_DISTAN
ERGONOMICS = LEFT_ERGONO + RIGHT_ERGONO
PREFER_EDGES = LEFT_EDGE_B + RIGHT_EDGE_B

FINGER_ID = LEFT_FINGID + RIGHT_FINGID

# Real data on w.p.m. for each letter, normalized.
# Higher values is better (higher w.p.m.)
with open(file="typing_data/manual-typing-data_qwerty.json", mode="r") as f:
    data_qwerty = json.load(fp=f)
with open(file="typing_data/manual-typing-data_dvorak.json", mode="r") as f:
    data_dvorak = json.load(fp=f)
data_values = list(data_qwerty.values()) + list(data_dvorak.values())
mean_value = sum(data_values) / len(data_values)
data_combine = []
for dv, qw in zip(DVORAK, QWERTY):
    if dv in data_dvorak.keys() and qw in data_qwerty.keys():
        data_combine.append((data_dvorak[dv] + data_qwerty[qw]) / 2)
    elif dv in data_dvorak.keys() and qw not in data_qwerty.keys():
        data_combine.append(data_dvorak[dv])
    elif dv not in data_dvorak.keys() and qw in data_qwerty.keys():
        data_combine.append(data_qwerty[qw])
    else:
        # Fill missing data with the mean
        data_combine.append(mean_value)


class Individual(TypedDict):
    genome: str
    fitness: float


Population = list[Individual]


def render_keyboard(individual: Individual) -> str:
    layout = individual["genome"]
    fitness = individual["fitness"]
    """Prints the keyboard in a nice way"""
    return (
        f"______________  ________________\n"
        f" ` 1 2 3 4 5 6  7 8 9 0 " + " ".join(layout[15:17]) + " Back\n"
        f"Tab " + " ".join(layout[0:5]) + "  " + " ".join(layout[17:24]) + " \\\n"
        f"Caps " + " ".join(layout[5:10]) + "  " + " ".join(layout[24:30]) + " Enter\n"
        f"Shift "
        + " ".join(layout[10:15])
        + "  "
        + " ".join(layout[30:35])
        + " Shift\n"
        f"\nAbove keyboard has fitness of: {fitness}"
    )


# <<<< DO NOT MODIFY


def initialize_individual(genome: str, fitness: float) -> Individual:
    """
    Purpose:        Create one individual
    Parameters:     genome as string, fitness as integer (higher better)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a dict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    return {"genome": genome, "fitness": fitness}


def initialize_pop(example_genome: str, pop_size: int) -> Population:
    """
    Purpose:        Create population to evolve
    Parameters:     Goal string, population size as int
    User Input:     no
    Prints:         no
    Returns:        a population, as a list of Individuals
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    pop = []
    for i in range(0, pop_size):
        genome_list = list(example_genome)
        random.shuffle(genome_list)
        genome = "".join(genome_list)
        pop.append(initialize_individual(str(genome), 0))
    return pop


def recombine_pair(parent1: Individual, parent2: Individual) -> Population:
    """
    Purpose:        Recombine two parents to produce two children
    Parameters:     Two parents as Individuals
    User Input:     no
    Prints:         no
    Returns:        A population of size 2, the children
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    # start of left keyboard: 0
    # end of left keyboard: 14
    # start of right keyboard: 15
    # end of right keyboard: 34
    pair = []
    # construct genomes
    leftParent1 = parent1["genome"][:15]
    rightParent1 = parent1["genome"][15:]
    leftParent2 = parent2["genome"][:15]
    rightParent2 = parent2["genome"][15:]
    genome1 = leftParent1
    genome2 = leftParent2
    # first child takes left side of parent one keyboard and then adds any non duplicates from parent two starting on right side
    for c in rightParent2:
        if c not in set(genome1):
            genome1 = genome1 + c
    for c in leftParent2:
        if c not in set(genome1):
            genome1 = genome1 + c
    # second child takes left side of parent two keyboard and then adds any non duplicates from parent 1 starting on right side
    for c in rightParent1:
        if c not in set(genome2):
            genome2 = genome2 + c
    for c in leftParent1:
        if c not in set(genome2):
            genome2 = genome2 + c
    # initialize children
    pair.append(initialize_individual(genome1, 0))
    pair.append(initialize_individual(genome1, 0))
    return pair


def recombine_group(parents: Population, recombine_rate: float) -> Population:
    """
    Purpose:        Recombines a whole group, returns the new population
                    Pair parents 1-2, 2-3, 3-4, etc..
                    Recombine at rate, else clone the parents.
    Parameters:     parents and recombine rate
    User Input:     no
    Prints:         no
    Returns:        New population of children
    Modifies:       Nothing
    Calls:          ?
    """
    index = 0
    pop = []
    while index < len(parents) - 1:
        if random.random() < recombine_rate:
            pop.append(recombine_pair(parents[index], parents[index + 1])[0])
            pop.append(recombine_pair(parents[index], parents[index + 1])[1])
        else:
            pop.append(parents[index])
            pop.append(parents[index + 1])
        index += 2
    return pop


def mutate_individual(parent: Individual, mutate_rate: float) -> Individual:
    """
    Purpose:        Mutate one individual
    Parameters:     One parents as Individual, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    temp_genome = list(parent["genome"])
    if random.random() < mutate_rate:
        position1 = random.randrange(0, len(temp_genome))
        position2 = random.randrange(0, len(temp_genome))
        while position2 == position1:
            position2 = random.randrange(0, len(temp_genome))
        temp_genome[position1], temp_genome[position2] = (
            temp_genome[position2],
            temp_genome[position1],
        )  # swap
    parent["genome"] = "".join(temp_genome)
    return parent


def mutate_group(children: Population, mutate_rate: float) -> Population:
    """
    Purpose:        Mutates a whole Population, returns the mutated group
    Parameters:     Population, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    for child in children:
        mutate_individual(child, mutate_rate)
    return children


# DO NOT MODIFY >>>>
def evaluate_individual(individual: Individual) -> None:
    """
    Purpose:        Computes and modifies the fitness for one individual
                    Assumes and relies on the logc of ./corpus/2_counts.py
    Parameters:     One Individual
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The individual (mutable object)
    Calls:          Basic python only
    Example doctest:
    """
    layout = individual["genome"]

    # This checks if the keyboard is valid.
    # It is a permutation of the given keys, without duplicates or deletions.
    if set(dvorak["genome"]) != set(layout) or len(dvorak["genome"]) != len(layout):
        individual["fitness"] = np.inf
        return

    fitness = 0

    # Basic return to home row, with no differential cost for repeats.
    for pos, key in enumerate(layout):
        fitness += count_dict[key] * int(DISTANCE[pos])

    # Top-down guess at ideal ergonomics
    for pos, key in enumerate(layout):
        fitness += count_dict[key] * int(ERGONOMICS[pos])

    # Keybr.com querty-dvorak average data as estimate of real hand
    for pos, key in enumerate(layout):
        # These float numbers range from (0-1],
        # transformed so higher floats are better/faster:
        fitness += 2 * (count_dict[key] / data_combine[pos])
        # 2 * was just to increase the emphasis on this real data a bit

    # Symbols should be toward edges.
    for pos, key in enumerate(layout):
        if key in "-[],.';/=":
            fitness += int(PREFER_EDGES[pos])

    # Vowels on the left, Consosants on the right
    for pos, key in enumerate(layout):
        if key in "AEIOUY" and pos > 14:
            fitness += 3

    # [] {} () <> should be adjacent.
    # () are fixed by design choice (number line).
    # [] and {} are on same keys.
    # Perhaps ideally, <> and () should be on same keys too...
    right_edges = [4, 9, 14, 16, 23, 29, 34]
    for pos, key in enumerate(layout):
        # order of (x or y) protects index on far right:
        if key == "[" and (pos in right_edges or "]" != layout[pos + 1]):
            fitness += 1
        if key == "," and (pos in right_edges or "." != layout[pos + 1]):
            fitness += 1

    # High transitional probabilities should be rolls
    # For example, 2-char sequences: in, ch, th, re, er, etc.
    # Rolls are rewarded inwards on the hand.
    # Left is left to right, and right is right to left.
    # left_edges = [0, 5, 10, 15, 17, 24, 30]
    # This is the left half of keyboard:
    for pos in range(len(layout) - 1):
        if pos in right_edges:
            continue
        char1 = layout[pos]
        char2 = layout[pos + 1]
        dict_key = char1 + char2
        # This is the right half of keyboard
        if pos > 14:
            char1, char2 = char2, char1
        fitness -= count_run2_dict[dict_key]

    # Penalize any 2 character run that occurs on the same finger,
    # in proportion to the count of the run.
    # If they don't occur on the same finger, no penalty.
    for keypair, freq in count_run2_dict.items():
        key1pos = layout.find(keypair[0])
        key2pos = layout.find(keypair[1])
        if FINGER_ID[key1pos] == FINGER_ID[key2pos]:
            fitness += freq

    # TODO Can you think of any more constraints to add to the fitness function?

    individual["fitness"] = fitness


# <<<< DO NOT MODIFY


def evaluate_group(individuals: Population) -> None:
    """
    Purpose:        Computes and modifies the fitness for population
    Parameters:     Objective string, Population
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The Individuals, all mutable objects
    Calls:          ?
    Example doctest:
    """
    for individual in individuals:
        evaluate_individual(individual)
        # print(individual)


def rank_group(individuals: Population) -> None:
    """
    Purpose:        Create one individual
    Parameters:     Population of Individuals
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The population's order (a mutable object)
    Calls:          ?
    Example doctest:
    """
    individuals.sort(key=lambda ind: ind["fitness"], reverse=False)


def parent_select(individuals: Population, number: int) -> Population:
    """
    Purpose:        Choose parents in direct probability to their fitness
    Parameters:     Population, the number of individuals to pick.
    User Input:     no
    Prints:         no
    Returns:        Sub-population
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    weights = []
    for individual in individuals:
        weights.append(individual["fitness"])
    # print(weights)
    return random.choices(individuals, weights, cum_weights=None, k=number)


def survivor_select(individuals: Population, pop_size: int) -> Population:
    """
    Purpose:        Picks who gets to live!
    Parameters:     Population, and population size to return.
    User Input:     no
    Prints:         no
    Returns:        Population, of pop_size
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    survivors = []
    for i in range(pop_size):
        survivors.append(individuals[i])
    return survivors


def evolve(example_genome: str, pop_size: int = 100) -> Population:
    """
    Purpose:        A whole EC run, main driver
    Parameters:     The evolved population of solutions
    User Input:     No
    Prints:         Updates every time fitness switches.
    Returns:        Population
    Modifies:       Various data structures
    Calls:          Basic python, all your functions
    """
    # To debug doctest test in pudb
    # Highlight the line of code below below
    # Type 't' to jump 'to' it
    # Type 's' to 'step' deeper
    # Type 'n' to 'next' over
    # Type 'f' or 'r' to finish/return a function call and go back to caller
    dvorak = initialize_individual(example_genome, 0)
    evaluate_individual(dvorak)
    population = initialize_pop(example_genome, pop_size)
    evaluate_group(population)
    rank_group(population)
    best_fitness = population[0]["fitness"]
    counter = 0
    while best_fitness > 68 and counter < 1000:
        counter += 1
        parents = parent_select(population, 80)
        children = recombine_group(parents=parents, recombine_rate=0.8)
        mutate_rate = 0.1
        mutants = mutate_group(children=children, mutate_rate=mutate_rate)
        evaluate_group(individuals=mutants)
        everyone = population + mutants
        rank_group(individuals=everyone)
        population = survivor_select(individuals=everyone, pop_size=pop_size)
        if best_fitness != population[0]["fitness"]:
            best_fitness = population[0]["fitness"]
            print("Iteration number", counter, "with best individual", population[0])
    return population


seed = False

# DO NOT MODIFY >>>>
if __name__ == "__main__":
    divider = "===================================================="
    # Execute doctests to protect main:
    # import doctest

    # doctest.testmod()
    # doctest.testmod(verbose=True)

    if seed:
        random.seed(42)

    with open("corpus/counts.json") as fhand:
        count_dict = json.load(fhand)
    # print({k: v for k, v in sorted(count_dict.items(), key=lambda item: item[1], reverse=True)})
    # print("Above is the order of frequency of letters in English.")

    # print("Counts of characters in big corpus, ordered by freqency:")
    # ordered = sorted(count_dict, key=count_dict.__getitem__, reverse=True)
    # for key in ordered:
    #     print(key, count_dict[key])

    with open("corpus/counts_run2.json") as fhand:
        count_run2_dict = json.load(fhand)
    # print({k: v for k, v in sorted(count_run2_dict.items(), key=lambda item: item[1], reverse=True)})
    # print("Above is the order of frequency of letter-pairs in English.")

    print(divider)
    print(
        f"Number of possible permutations of standard keyboard: {math.factorial(len(DVORAK)):,e}"
    )
    print("That's a huge space to search through")
    print("The messy landscape is a difficult to optimize multi-modal space")
    print("Lower fitness is better.")

    print(divider)
    print("\nThis is the Dvorak keyboard:")
    dvorak = Individual(genome=DVORAK, fitness=0)
    evaluate_individual(dvorak)
    print(render_keyboard(dvorak))

    print(divider)
    print("\nThis is the Workman keyboard:")
    workman = Individual(genome=WORKMAN, fitness=0)
    evaluate_individual(workman)
    print(render_keyboard(workman))

    print(divider)
    print("\nThis is the Colemak keyboard:")
    colemak = Individual(genome=COLEMAK, fitness=0)
    evaluate_individual(colemak)
    print(render_keyboard(colemak))

    print(divider)
    print("\nThis is the QWERTY keyboard:")
    qwerty = Individual(genome=QWERTY, fitness=0)
    evaluate_individual(qwerty)
    print(render_keyboard(qwerty))

    print(divider)
    print("\nThis is a random layout:")
    badarr = list(DVORAK)
    random.shuffle(badarr)
    badstr = "".join(badarr)
    badkey = Individual(genome=badstr, fitness=0)
    evaluate_individual(badkey)
    print(render_keyboard(badkey))

    print(divider)
    print("Below, we print parts of the fitness map (not keyboards themselves)")

    print("\n\nThis is the distance assumption:")
    dist = Individual(genome=DISTANCE, fitness=0)
    print(render_keyboard(dist))

    print("\n\nThis is another human-invented ergonomics assumption:")
    ergo = Individual(genome=ERGONOMICS, fitness=0)
    print(render_keyboard(ergo))

    print("\n\nThis is the edge-avoidance mechanism for special characters:")
    edges = Individual(genome=PREFER_EDGES, fitness=0)
    print(render_keyboard(edges))

    print("\n\nThis is real typing data, transformed so bigger=better:")
    realdata = "".join(
        [str(int(round(reaction_time * 10, 0) - 1)) for reaction_time in data_combine]
    )
    real_rt = Individual(genome=realdata, fitness=0)
    print(render_keyboard(real_rt))

    print("\n\nThis is the finger typing map:")
    edges = Individual(genome=FINGER_ID, fitness=0)
    print(render_keyboard(edges))

    print(divider)
    input("Press any key to start")
    population = evolve(example_genome=DVORAK)

    print("Here is the best layout:")
    print(render_keyboard(population[0]))

    grade = 0
    if qwerty["fitness"] < population[0]["fitness"]:
        grade = 0
    elif colemak["fitness"] < population[0]["fitness"]:
        grade = 50
    elif workman["fitness"] < population[0]["fitness"]:
        grade = 60
    elif dvorak["fitness"] < population[0]["fitness"]:
        grade = 70
    else:
        grade = 80

    with open(file="results.txt", mode="w") as f:
        f.write(str(grade))

    with open(file="best.json", mode="w") as f:
        f.write(json.dumps(population[0]))

    with open(file="best.txt", mode="w") as f:
        f.write(render_keyboard(population[0]))
# <<<< DO NOT MODIFY
