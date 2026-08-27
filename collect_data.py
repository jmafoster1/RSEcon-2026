import random
from multiprocessing import Pool

import covasim as cv
import networkx as nx
import numpy as np
import pandas as pd
from causal_testing.discovery.hill_climber_discovery import HillClimberDiscovery

random.seed(0)
np.random.seed(0)


def cumulative_infections(sim: cv.Sim):
    return int(sim.results["cum_infections"][-1])


def single_run(beta: float, location: str):
    sim = cv.Sim(beta=beta, location=location, pop_type="hybrid", verbose=0)
    sim.run()
    return {
        "location": sim.pars["location"],
        "beta": sim.pars["beta"],
        "average_age": sim.people["age"].mean(),
        "contacts_home": sim.pars["contacts"]["h"],
        "contacts_school": sim.pars["contacts"]["s"],
        "contacts_work": sim.pars["contacts"]["w"],
        "contacts_community": sim.pars["contacts"]["c"],
        "cum_infections": cumulative_infections(sim),
    }


RUNS = 10000
betas = np.linspace(0.010, 0.020, RUNS)  # Sweep beta from 0.01 to 0.02
locations = np.random.choice(list(cv.data.country_age_data.data), size=RUNS)

with Pool(10) as pool:
    data = pool.starmap(single_run, zip(betas, locations))
big_data = pd.DataFrame(data)
big_data.to_csv("big_data.csv")

hill_climber = HillClimberDiscovery(
    df=big_data,
    exclude_edges=[(".*", "beta"), (".*", "location"), ("cum_infections", ".*")],
)
discovered_dag = hill_climber.discover()
nx.nx_pydot.write_dot(discovered_dag, "discovered_dag.dot")
