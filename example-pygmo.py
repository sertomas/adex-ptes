from tespy.networks import Network
from tespy.components import Condenser
from tespy.components import CycleCloser
from tespy.components import SimpleHeatExchanger
from tespy.components import Pump
from tespy.components import Sink
from tespy.components import Source
from CoolProp.CoolProp import PropsSI as PSI
from tespy.connections import Connection
import numpy as np

import pygmo as pg
from tespy.tools.optimization import OptimizationProblem


class SamplePlant:
    """Class template for TESPy model usage in optimization module."""
    def __init__(self):
        
        working_fluid = "NH3"
        self.network = Network()
        self.network.set_attr(
            p_unit="bar", T_unit="C", h_unit="kJ / kg", iterinfo=False
        )
        # sources & sinks
        c_in = Source("refrigerant in")
        cons_closer = CycleCloser("consumer cycle closer")
        va = Sink("valve")
        
        # consumer system
        cd = Condenser("condenser")
        rp = Pump("pump")
        cons = SimpleHeatExchanger("consumer")

        c0 = Connection(c_in, "out1", cd, "in1", label="0")
        c1 = Connection(cd, "out1", va, "in1", label="1")
        
        c20 = Connection(cons_closer, "out1", rp, "in1", label="20")
        c21 = Connection(rp, "out1", cd, "in2", label="21")
        c22 = Connection(cd, "out2", cons, "in1", label="22")
        c23 = Connection(cons, "out1", cons_closer, "in1", label="23")
        
        self.network.add_conns(c0, c1, c20, c21, c22, c23)
        
        cd.set_attr(pr1=0.99, pr2=0.99)
        rp.set_attr(eta_s=0.75)
        cons.set_attr(pr=0.99)
        
        p_cond = PSI("P", "Q", 1, "T", 273.15 + 95, working_fluid) / 1e5
        c0.set_attr(T=170, p=p_cond, fluid={working_fluid: 1})
        c20.set_attr(T=60, p=2, fluid={"water": 1})
        c22.set_attr(T=90)
        
        # key design paramter
        cons.set_attr(Q=-230e3)
        
        self.network.solve("design")
        self.network.print_results()
        self.solved = True

    # %%[sec_2]

    def get_param(self, obj, label, parameter):
        """Get the value of a parameter in the network"s unit system.

        Parameters
        ----------
        obj : str
            Object to get parameter for (Components/Connections).

        label : str
            Label of the object in the TESPy model.

        parameter : str
            Name of the parameter of the object.

        Returns
        -------
        value : float
            Value of the parameter.
        """
        if obj == "Components":
            return self.network.get_comp(label).get_attr(parameter).val
        elif obj == "Connections":
            return self.network.get_conn(label).get_attr(parameter).val

    def set_params(self, **kwargs):

        if "Connections" in kwargs:
            for c, params in kwargs["Connections"].items():
                self.network.get_conn(c).set_attr(**params)

        if "Components" in kwargs:
            for c, params in kwargs["Components"].items():
                self.network.get_comp(c).set_attr(**params)

    def solve_model(self, **kwargs):
        """
        Solve the TESPy model given the the input parameters
        """
        self.set_params(**kwargs)

        self.solved = False
        try:
            self.network.solve("design")
            if not self.network.converged:
                self.network.solve("design", init_only=True)
            else:
                # might need more checks here!
                if (
                        any(self.network.results["condenser"]["Q"] > 0)
                        or any(self.network.results["pump"]["P"] < 0)
                ):
                    self.solved = False
                else:
                    self.solved = True
        except ValueError as e:
            self.network.lin_dep = True
            self.network.solve("design", init_only=True)

    def get_objective(self, objective=None):
        """
        Get the current objective function evaluation.

        Parameters
        ----------
        objective : str
            Name of the objective function.

        Returns
        -------
        objective_value : float
            Evaluation of the objective function.
        """
        if self.solved:
            if objective == "efficiency":
                return self.get_param('Connections', '21', 'p')
            else:
                msg = f"Objective {objective} not implemented."
                raise NotImplementedError(msg)
        else:
            return np.nan


plant = SamplePlant()

plant.get_objective("efficiency")
variables = {
    "Connections": {
        "2": {"p": {"min": 1, "max": 40}},
        "4": {"p": {"min": 1, "max": 40}}
    }
}
constraints = {
    "lower limits": {
        "Connections": {
            "2": {"p": "ref1"}
        },
    },
    "ref1": ["Connections", "4", "p"]
}

optimize = OptimizationProblem(
    plant, variables, constraints, objective="efficiency"
)
# %%[sec_4]
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
# %%[sec_5]
# To access the results
print(optimize.individuals)