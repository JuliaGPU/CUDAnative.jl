module GeneticAlgorithm

# This benchmark runs a genetic algorithm on the GPU.
# The population is stored in linked lists and characters
# are stored in heap memory.

using CUDAnative, CUDAdrv
import ..LinkedList: List, Nil, Cons, foldl, map, max
import ..CUDArandom: LinearCongruentialGenerator, next

# A character in our genetic algorithm, based loosely on Fallout's SPECIAL system.
mutable struct Character
    strength::Int
    perception::Int
    endurance::Int
    charisma::Int
    intelligence::Int
    agility::Int
    luck::Int
end

# Computes the mean of two integers.
function mean(a::Int, b::Int)::Int
    div(a + b, 2)
end

function crossover(parent_one::Character, parent_two::Character)::Character
    Character(
        mean(parent_one.strength, parent_two.strength),
        mean(parent_one.perception, parent_two.perception),
        mean(parent_one.endurance, parent_two.endurance),
        mean(parent_one.charisma, parent_two.charisma),
        mean(parent_one.intelligence, parent_two.intelligence),
        mean(parent_one.agility, parent_two.agility),
        mean(parent_one.luck, parent_two.luck))
end

function mutate_stat(value::Int, generator::LinearCongruentialGenerator)::Int
    new_stat = value + next(generator, -2, 3)
    if new_stat > 10
        return 10
    elseif new_stat < 0
        return 0
    else
        return new_stat
    end
end

function mutate(original::Character, generator::LinearCongruentialGenerator)::Character
    Character(
        mutate_stat(original.strength, generator),
        mutate_stat(original.perception, generator),
        mutate_stat(original.endurance, generator),
        mutate_stat(original.charisma, generator),
        mutate_stat(original.intelligence, generator),
        mutate_stat(original.agility, generator),
        mutate_stat(original.luck, generator))
end

function random_character(generator::LinearCongruentialGenerator)::Character
    Character(
        next(generator, 0, 11),
        next(generator, 0, 11),
        next(generator, 0, 11),
        next(generator, 0, 11),
        next(generator, 0, 11),
        next(generator, 0, 11),
        next(generator, 0, 11))
end

# Computes the fitness of a character.
function fitness(individual::Character)::Float64
    # Compute the character's cost, i.e., the sum of their stats.
    cost = Float64(individual.strength
        + individual.perception
        + individual.endurance
        + individual.charisma
        + individual.intelligence
        + individual.agility
        + individual.luck)

    # Compute the character's true fitness, i.e., how well we expect
    # the character to perform.
    true_fitness = 0.0

    function stat_fitness(stat::Int)::Float64
        if stat >= 5
            # Linear returns for stats greater than five.
            return Float64(stat)
        else
            # Very low stats make for a poor character build.
            return Float64(stat * stat) / 25.0
        end
    end

    # Evaluate stats.
    true_fitness += stat_fitness(individual.strength)
    true_fitness += stat_fitness(individual.perception)
    true_fitness += stat_fitness(individual.endurance)
    true_fitness += stat_fitness(individual.charisma)
    true_fitness += stat_fitness(individual.intelligence)
    true_fitness += stat_fitness(individual.agility)
    true_fitness += stat_fitness(individual.luck)

    # We like charisma, intelligence and luck.
    true_fitness += Float64(individual.charisma)
    true_fitness += Float64(individual.intelligence)
    true_fitness += Float64(individual.luck)

    true_fitness - cost + 100.0
end

function fittest(population::List{Character})::Character
    max(fitness, population, Character(0, 0, 0, 0, 0, 0, 0))
end

function step(population::List{Character}, generator::LinearCongruentialGenerator)::List{Character}
    # Find the fittest individual in the population.
    best = fittest(population)
    # Do a bunch of crossovers and mutate the resulting population.
    map(x -> mutate(crossover(best, x), generator), population)
end

function genetic_algo(seed::Int)::Character
    generator = LinearCongruentialGenerator(seed)

    # Generate some random characters.
    individuals = Nil{Character}()
    for j in 1:10
        individuals = Cons{Character}(random_character(generator), individuals)
    end

    # Run the genetic algorithm for a few iterations.
    for j in 1:2
        individuals = step(individuals, generator)
    end

    # Find the best individual in the population.
    fittest(individuals)
end

const thread_count = 256

function kernel(results::CUDAnative.DevicePtr{Float64})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    fittest_individual = genetic_algo(i)
    unsafe_store!(results, fitness(fittest_individual), i)
end

end

function genetic_benchmark()
    destination_array = Mem.alloc(Mem.DeviceBuffer, sizeof(Float64) * GeneticAlgorithm.thread_count)
    destination_pointer = Base.unsafe_convert(CuPtr{Float64}, destination_array)
    @cuda_sync threads=GeneticAlgorithm.thread_count GeneticAlgorithm.kernel(destination_pointer)
end

@cuda_benchmark "genetic algo" genetic_benchmark()
