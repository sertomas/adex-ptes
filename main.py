"""
Author: Sergio Tomasinelli
Date: December 2023
Description: Modeling, simulation and advanced exergy analysis of a Carnot Battery
"""

import json
from func_var import run_hp, run_orc, perf_adex_hp, perf_adex_orc
from tespy.tools import logger
import logging
logger.define_logging(screen_level=logging.WARNING, file_level=logging.INFO)


with open('inputs/config.json') as file:
    config = json.load(file)
with open('inputs/hp_pars.json') as file:
    hp_pars_base = json.load(file)
with open('inputs/orc_pars.json') as file:
    orc_pars_base = json.load(file)

# --- RUN MODELS -------------------------------------------------------------------------------------------------------

hp_base = run_hp(config, hp_pars_base, 'base', Tdeltah=True, logph=False, show=False)
hp_base_df = hp_base.network.results['Connection']
orc_base = run_orc(config, orc_pars_base, 'base', Tdeltah=True, logph=False, show=False)
orc_base_df = orc_base.network.results['Connection']


# --- PERFORM CONVENTIONAL EXERGY ANALYSIS -----------------------------------------------------------------------------

hp_base.perform_exergy_analysis(path='outputs/exan/hp_base_exan_')
print("Conventional exergy analysis of the HP base case: epsilon_real =", round(hp_base.epsilon, 3))

orc_base.perform_exergy_analysis(path='outputs/exan/orc_base_exan_')
print("Conventional exergy analysis of the ORC base case: epsilon_real =", round(orc_base.epsilon, 3))


# --- PERFORM ADVANCED EXERGY ANALYSIS ---------------------------------------------------------------------------------
perf_adex_hp(hp_base, p32=13.5, components=['COMP', 'COND', 'IHX', 'VAL', 'EVA'], check_temp_diff=True)

perf_adex_orc(orc_base, ttd_u_eva=9, ttd_l_cond=6.5, components=['EVA', 'EXP', 'IHX', 'COND', 'PUMP'], check_temp_diff=True)

