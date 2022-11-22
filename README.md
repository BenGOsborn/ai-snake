# AI Snake

An AI powered snake game featuring genetic algorithms and deep Q learning.

## Requirements

-   Python 3.8.10

`pip3 install -r requirements.txt`

## Instructions

### Running

Download and extract the [weights](https://github.com/BenGOsborn/ai-snake/releases) and drag them into the `bin` folder

#### Genetic algorithm

-   With display - `python3 main.py d ga`
-   Without display - `python3 main.py ga`

#### Deep Q network

-   With display - `python3 main.py d dqn`
-   Without display - `python3 main.py dqn`

### Training

**NOTE** training will overwrite any existing weights

-   Genetic algorithm - `python3 train.py ga`
-   Deep Q network - `python3 train.py ga`
