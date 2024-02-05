import numpy as np
from CoolProp.CoolProp import PropsSI as PSI
import pandas as pd

from func_fix import (pr_func, pr_deriv, turbo_func, turbo_deriv, he_func, he_deriv,
                      ihx_func, ihx_deriv, temperature_func, temperature_deriv, valve_func, valve_deriv,
                      x_saturation_func, x_saturation_deriv, eps_compressor_func, eps_compressor_deriv, eps_real_he_func,
                      eps_real_he_deriv)

T_0 = 283.15

epsilon = pd.read_csv("outputs/exan/hp_base_exan_components.csv", index_col=0)["epsilon"]

wf = "REFPROP::R1336MZZZ"
fluid_TES = "REFPROP::water"

# TES
t21 = 70 + 273.15    # input is known
p21 = 5 * 1e5        # input is known
t22 = 140 + 273.15   # output temperature is set
m21 = 10             # dimensioning of the system (full load)

# PRESSURE DROPS
pr_cond_cold = 1
pr_cond_hot = 1

# HEAT PUMP
ttd_l_COND = 0

# pre-calculation
t32 = t21 + ttd_l_COND

# STARTING VALUES
h31 = 506e3
p31 = 13.7e5
h32 = 287e3
m31 = 13.7
h21 = 295e3
h22 = 590e3
p22 = 4.9e5
h38 = 480e3
h39 = 370e3
h28 = 390e3
h29 = 540e3

variables = np.array([h31, h32, m31, h21, h22, h38, h39, h28, h29])
residual = np.ones(9)

iter = 0

t31 = 152.93+273.15
p32 = 12e5

while np.linalg.norm(residual) > 1e-3:
    # TODO [h31, h32, m31, h21, h22, h38, h39, h28, h29]
    # TODO   0    1    2    3    4    5    6    7    8
    t31_set = temperature_func(t31, variables[0], p31, wf)
    t21_set = temperature_func(t21, variables[3], p21, fluid_TES)
    t22_set = temperature_func(t22, variables[4], p22, fluid_TES)
    cond_tot_en_bal = he_func(variables[2], variables[0], variables[1], m21, variables[3], variables[4])
    eps_cond_def = eps_real_he_func(epsilon['Condenser'], variables[0], p31, variables[1], p32, variables[2], wf,
                                    variables[3], p21, variables[4], p22, m21, fluid_TES)
    cond_eco_outlet_sat = x_saturation_func(1, variables[5], p31, wf)
    cond_eco_en_bal = he_func(variables[2], variables[0], variables[5], m21, variables[8], variables[4])
    cond_sh_inlet_sat = x_saturation_func(0, variables[6], p31, wf)
    cond_sh_en_bal = he_func(variables[2], variables[6], variables[1], m21, variables[3], variables[7])

    residual = np.array([t31_set, t21_set, t22_set, cond_tot_en_bal, eps_cond_def, cond_eco_outlet_sat,
                         cond_eco_en_bal, cond_sh_inlet_sat, cond_sh_en_bal], dtype=float)
    jacobian = np.zeros((9, 9))
    
    t31_set_j = temperature_deriv(t31, variables[0], p31, wf)
    t21_set_j = temperature_deriv(t21, variables[3], p21, fluid_TES)
    t22_set_j = temperature_deriv(t22, variables[4], p22, fluid_TES)
    cond_tot_en_bal_j = he_deriv(variables[2], variables[0], variables[1], m21, variables[3], variables[4])
    eps_cond_def_j = eps_real_he_deriv(epsilon['Condenser'], variables[0], p31, variables[1], p32, variables[2], wf,
                                       variables[3], p21, variables[4], p22, m21, fluid_TES)
    cond_eco_outlet_sat_j = x_saturation_deriv(1, variables[5], p31, wf)
    cond_eco_en_bal_j = he_deriv(variables[2], variables[0], variables[5], m21, variables[8], variables[4])
    cond_sh_inlet_sat_j = x_saturation_deriv(0, variables[6], p31, wf)
    cond_sh_en_bal_j = he_deriv(variables[2], variables[6], variables[1], m21, variables[3], variables[7])
    
    jacobian[0, 0] = t31_set_j["h"]
    jacobian[1, 3] = t21_set_j["h"]
    jacobian[2, 4] = t22_set_j["h"]
    jacobian[3, 2] = cond_tot_en_bal_j["m_hot"]
    jacobian[3, 0] = cond_tot_en_bal_j["h_1"]
    jacobian[3, 1] = cond_tot_en_bal_j["h_2"]
    jacobian[3, 3] = cond_tot_en_bal_j["h_3"]
    jacobian[3, 4] = cond_tot_en_bal_j["h_4"]
    jacobian[4, 2] = eps_cond_def_j["m_hot"]
    jacobian[4, 0] = eps_cond_def_j["h_hot_in"]
    jacobian[4, 1] = eps_cond_def_j["h_hot_out"]
    jacobian[4, 3] = eps_cond_def_j["h_cold_in"]
    jacobian[4, 4] = eps_cond_def_j["h_cold_out"]
    jacobian[5, 5] = cond_eco_outlet_sat_j["h"]
    jacobian[6, 2] = cond_eco_en_bal_j["m_hot"]
    jacobian[6, 0] = cond_eco_en_bal_j["h_1"]
    jacobian[6, 5] = cond_eco_en_bal_j["h_2"]
    jacobian[6, 8] = cond_eco_en_bal_j["h_3"]
    jacobian[6, 4] = cond_eco_en_bal_j["h_4"]
    jacobian[7, 6] = cond_sh_inlet_sat_j["h"]  
    jacobian[8, 2] = cond_sh_en_bal_j["m_hot"]
    jacobian[8, 6] = cond_sh_en_bal_j["h_1"]
    jacobian[8, 1] = cond_sh_en_bal_j["h_2"]
    jacobian[8, 3] = cond_sh_en_bal_j["h_3"]
    jacobian[8, 7] = cond_sh_en_bal_j["h_4"]

    # Convert the numpy array to a pandas DataFrame
    df = pd.DataFrame(jacobian)

    # Save the DataFrame as a CSV file
    df.round(4).to_csv('jacobian_matrix.csv', index=False)

    variables -= np.linalg.inv(jacobian).dot(residual)

    cond_number = np.linalg.cond(jacobian)
    # print("Condition number: ", cond_number, " and residual: ", np.linalg.norm(residual))

    # if variables[15] < 0:
    #    variables[15] = -variables[15]

    iter += 1

print(f"Simulation converged successfully after {iter} iterations.")