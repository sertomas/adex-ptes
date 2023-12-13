import json
import pygmo as pg
import pandas as pd
from func_var import run_hp
from tespy.networks import Network
from tespy.tools import logger
from tespy.tools.optimization import OptimizationProblem
import numpy as np
import logging
logger.define_logging(screen_level=logging.WARNING, file_level=logging.INFO)


with open('inputs/config.json') as file:
    config = json.load(file)
with open('inputs/hp_pars.json') as file:
    hp_pars_base = json.load(file)


hp_base = run_hp(config, hp_pars_base, 'base', Tdeltah=False, logph=False, show=False)

# this is how you read a parameter
# print(hp_base.get_param('Connections', '32', 'p'))
# this is how you set a parameter
# hp_base.set_params(Connections={'32': {'p': 14}})

hp_base.get_objective("p32")
variables = {
    "Components": {
        "Evaporator": {"ttd_u": {"min": 5, "max": 10}}
    }
}
constraints = {
    "lower limits": {
        "Components": {
            "Evaporator": {"ttd_u": "ref1"}
        },
    },
    "ref1": ["Components", "Evaporator", "ttd_u"]
}

optimize = OptimizationProblem(
    hp_base, variables, constraints, objective="p32"
)

num_ind = 10
num_gen = 100

# for algorithm selection and parametrization please consider the pygmo
# documentation! The number of generations indicated in the algorithm is
# the number of evolutions we undertake within each generation defined in
# num_gen
algo = pg.algorithm(pg.ihs(gen=3, seed=42))
# create starting population
pop = pg.population(pg.problem(optimize), size=num_ind, seed=42)
optimize.run(algo, pop, num_ind, num_gen)