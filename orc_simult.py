import numpy as np
from CoolProp.CoolProp import PropsSI as PSI
import pandas as pd
from func_fix import (pr_func, pr_deriv, eta_s_PUMP_func, eta_s_PUMP_deriv, turbo_func, turbo_deriv, he_func, he_deriv,
                         ihx_func, ihx_deriv, temperature_func, temperature_deriv, valve_func, valve_deriv,
                         x_saturation_func, x_saturation_deriv, ttd_func, ttd_deriv, eta_s_EXP_func, eta_s_EXP_deriv)

wf = "REFPROP::R134a"
fluid_TES = "REFPROP::water"
fluid_ambient = "REFPROP::air"

# TES
t51 = 140 + 273.15  # input is known
p51 = 5 * 1e5       # input is known
t52 = 70 + 273.15   # output temperature is set
m51 = 10            # dimensioning of the system (full load)

# PRESSURE DROPS
pr_cond_cold = 1
pr_cond_hot = 0.95
pr_ihx_hot = 0.985
pr_ihx_cold = 0.985
pr_eva_cold = 1
pr_eva_hot = 0.95

# AMBIENT
t41 = 10 + 273.15  # input is known
p41 = 1.013 * 1e5  # input is known
t42 = 13 + 273.15  # output temperature is set

# HEAT PUMP
ttd_l_COND = 6.5  # design variable to optimize?
ttd_l_IHX = 5
ttd_u_EVA = 8.5   # design variable to optimize?
p62 = 50 * 1e5  # variable?

# TECHNICAL PARAMETERS
# eta_s = 0.844949  # --> it seems to be different from TESPy results (look T_31)
eta_s_PUMP = 0.85
eta_s_EXP = 0.9

# pre-calculation
t61 = t41 + ttd_l_COND
t64 = t51 - ttd_u_EVA

# STARTING VALUES
h61 = 220e3
h62 = 220e3
h63 = 240e3
h64 = 475e3
h65 = 425e3
h66 = 410e3
m61 = 12.8
h51 = 600e3
h52 = 300e3
h41 = 415e3
h42 = 400e3
m41 = 800
P_EXP = 540e3
P_PUMP = 200e3
p61 = 5 * 1e5
p63 = 49 * 1e5
p64 = 46 * 1e5
p65 = 5.5 * 1e5
p66 = 5.6 * 1e5
p42 = 1 * 1e5
p52 = 5 * 1e5

variables = np.array([h41, h42, p42, h61, p61, h62, h66, p66, p63, p64, h64, p65, h65, h63, h51, h52, p52, m61, m41,
                      P_EXP, P_PUMP])
residual = np.ones(len(variables))

iter = 0

while np.linalg.norm(residual) > 1e-4:
    # TODO [h41, h42, p42, h61, p61, h62, h66, p66, p63, p64, h64, p65, h65, h63, h51, h52, p52, m61, m41, P_EXP, P_PUMP]
    # TODO   0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18    19     20
    t41_set = temperature_func(t41, variables[0], p41, fluid_ambient)
    t42_set = temperature_func(t42, variables[1], variables[2], fluid_ambient)
    p42_set = pr_func(pr_cond_cold, p41, variables[2])
    t61_set = temperature_func(t61, variables[3], variables[4], wf)
    cond_outlet_sat = x_saturation_func(0, variables[3], variables[4], wf)
    eta_s_PUMP_def = eta_s_PUMP_func(eta_s_PUMP, variables[3], variables[4], variables[5], p62, wf)
    t66_ttd = ttd_func(variables[5], p62, ttd_l_IHX, wf)
    t66_set = temperature_func(t66_ttd, variables[6], variables[7], wf)
    p66_set = pr_func(pr_cond_hot, variables[7], variables[4])
    p63_set = pr_func(pr_ihx_cold, p62, variables[8])
    p64_set = pr_func(pr_eva_hot, variables[8], variables[9])
    t64_set = temperature_func(t64, variables[10], variables[9], wf)
    p65_set = pr_func(pr_ihx_hot, variables[11], variables[7])
    eta_s_EXP_def = eta_s_EXP_func(eta_s_EXP, variables[10], variables[9], variables[12], variables[11], wf)
    ihx_en_bal = ihx_func(variables[12], variables[6], variables[5], variables[13])
    t51_set = temperature_func(t51, variables[14], p51, fluid_TES)
    t52_set = temperature_func(t52, variables[15], variables[16], fluid_TES)
    p52_set = pr_func(pr_cond_hot, variables[15], variables[16])
    cond_en_bal = he_func(m51, variables[14], variables[15], variables[17], variables[13], variables[10])
    eva_en_bal = he_func(variables[17], variables[6], variables[3], variables[18], variables[0], variables[1])
    exp_en_bal = turbo_func(variables[19], variables[17], variables[10], variables[12])
    pump_en_bal = turbo_func(variables[20], variables[17], variables[3], variables[5])

    residual = np.array([t41_set, t42_set, p42_set, t61_set, cond_outlet_sat, eta_s_PUMP_def, t66_set, p66_set,
                         p63_set, p64_set, t64_set, p65_set, eta_s_EXP_def, ihx_en_bal, t51_set, t52_set, p52_set,
                         cond_en_bal, eva_en_bal, exp_en_bal, pump_en_bal])
    jacobian = np.zeros((len(variables), len(variables)))

    t41_set_j = temperature_deriv(t41, variables[0], p41, fluid_ambient)
    t42_set_j = temperature_deriv(t42, variables[1], variables[2], fluid_ambient)
    p42_set_j = pr_deriv(pr_cond_cold, p41, variables[2])
    t61_set_j = temperature_deriv(t61, variables[3], p61, wf)
    cond_outlet_sat_j = x_saturation_deriv(0, variables[3], variables[4], wf)
    eta_s_PUMP_def_j = eta_s_PUMP_deriv(eta_s_PUMP, variables[3], variables[4], variables[5], p62, wf)
    t66_set_j = temperature_deriv(t66_ttd, variables[6], variables[7], wf)
    p66_set_j = pr_deriv(pr_eva_hot, variables[7], variables[4])
    p63_set_j = pr_deriv(pr_ihx_cold, p62, variables[8])
    p64_set_j = pr_deriv(pr_cond_cold, variables[8], variables[9])
    t64_set_j = temperature_deriv(t64, variables[10], variables[9], wf)
    p65_set_j = pr_deriv(1, variables[11], variables[7])
    eta_s_EXP_def_j = eta_s_EXP_deriv(eta_s_EXP, variables[10], variables[9], variables[12], variables[11], wf)
    ihx_en_bal_j = ihx_deriv(variables[12], variables[6], variables[5], variables[13])
    t51_set_j = temperature_deriv(t51, variables[14], p51, fluid_TES)
    t52_set_j = temperature_deriv(t52, variables[15], variables[16], fluid_TES)
    p52_set_j = pr_deriv(pr_cond_hot, variables[15], variables[16])
    cond_en_bal_j = he_deriv(m51, variables[14], variables[15], variables[17], variables[13], variables[10])
    eva_en_bal_j = he_deriv(variables[17], variables[6], variables[3], variables[18], variables[0], variables[1])
    exp_en_bal_j = turbo_deriv(variables[19], variables[17], variables[10], variables[12])
    pump_en_bal_j = turbo_deriv(variables[20], variables[17], variables[3], variables[5])

    # TODO [h41, h42, p42, h61, p61, h62, h66, p66, p63, p64, h64, p65, h65, h63, h51, h52, p52, m61, m41, P_EXP, P_PUMP]
    # TODO   0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18    19     20
    jacobian[0, 0] = t41_set_j["h"]  # derivative of t41_set with respect to h41
    jacobian[1, 1] = t42_set_j["h"]  # derivative of t42_set with respect to h42
    jacobian[1, 2] = t42_set_j["p"]  # derivative of t42_set with respect to p42
    jacobian[2, 2] = p42_set_j["p_2"]  # derivative of p42_set with respect to p42
    jacobian[3, 3] = t61_set_j["h"]  # derivative of t61_set with respect to h61
    jacobian[3, 4] = t61_set_j["p"]  # derivative of t61_set with respect to p61
    jacobian[4, 3] = cond_outlet_sat_j["h"]  # derivative of cond_outlet_sat with respect to h61
    jacobian[4, 4] = cond_outlet_sat_j["p"]  # derivative of cond_outlet_sat with respect to p61
    jacobian[5, 3] = eta_s_PUMP_def_j["h_1"]  # derivative of eta_s_PUMP_def with respect to h61
    jacobian[5, 4] = eta_s_PUMP_def_j["p_1"]  # derivative of eta_s_PUMP_def with respect to p61
    jacobian[5, 5] = eta_s_PUMP_def_j["h_2"]  # derivative of eta_s_PUMP_def with respect to h62
    jacobian[6, 6] = t66_set_j["h"]  # derivative of t66_set with respect to h66
    jacobian[6, 7] = t66_set_j["p"]  # derivative of t66_set with respect to p66
    jacobian[7, 7] = p66_set_j["p_1"]  # derivative of p66_set with respect to p66
    jacobian[7, 4] = p66_set_j["p_2"]  # derivative of p66_set with respect to p61
    jacobian[8, 8] = p63_set_j["p_2"]  # derivative of p63_set with respect to p63
    jacobian[9, 8] = p64_set_j["p_1"]  # derivative of p64_set with respect to p63
    jacobian[9, 9] = p64_set_j["p_2"]  # derivative of p64_set with respect to p64
    jacobian[10, 10] = t64_set_j["h"]  # derivative of t64_set with respect to h64
    jacobian[10, 9] = t64_set_j["p"]  # derivative of t64_set with respect to p64
    jacobian[11, 11] = p65_set_j["p_1"]  # derivative of p64_set with respect to p65
    jacobian[11, 7] = p65_set_j["p_2"]  # derivative of p64_set with respect to p66
    jacobian[12, 10] = eta_s_EXP_def_j["h_1"]  # derivative of eta_s_EXP_def with respect to h64
    jacobian[12, 9] = eta_s_EXP_def_j["p_1"]  # derivative of eta_s_EXP_def with respect to p64
    jacobian[12, 12] = eta_s_EXP_def_j["h_2"]  # derivative of eta_s_EXP_def with respect to h65
    jacobian[12, 11] = eta_s_EXP_def_j["p_2"]  # derivative of eta_s_EXP_def with respect to p65
    jacobian[13, 12] = ihx_en_bal_j["h_1"]  # derivative of ihx_en_bal with respect to h65
    jacobian[13, 6] = ihx_en_bal_j["h_2"]  # derivative of ihx_en_bal with respect to h66
    jacobian[13, 5] = ihx_en_bal_j["h_3"]  # derivative of ihx_en_bal with respect to h62
    jacobian[13, 13] = ihx_en_bal_j["h_4"]  # derivative of ihx_en_bal with respect to h63
    jacobian[14, 14] = t51_set_j["h"]  # derivative of t51_set with respect to h51
    jacobian[15, 15] = t52_set_j["h"]  # derivative of t52_set with respect to h52
    jacobian[15, 16] = t52_set_j["p"]  # derivative of t52_set with respect to p52
    jacobian[16, 16] = p52_set_j["p_2"]  # derivative of p52_set with respect to p52
    jacobian[17, 14] = cond_en_bal_j["h_1"]  # derivative of cond_en_bal with respect to h51
    jacobian[17, 15] = cond_en_bal_j["h_2"]  # derivative of cond_en_bal with respect to h52
    jacobian[17, 17] = cond_en_bal_j["m_cold"]  # derivative of cond_en_bal with respect to m61
    jacobian[17, 13] = cond_en_bal_j["h_3"]  # derivative of cond_en_bal with respect to h63
    jacobian[17, 10] = cond_en_bal_j["h_4"]  # derivative of cond_en_bal with respect to h64
    jacobian[18, 17] = eva_en_bal_j["m_hot"]  # derivative of eva_en_bal with respect to m41
    jacobian[18, 6] = eva_en_bal_j["h_1"]  # derivative of eva_en_bal with respect to h56
    jacobian[18, 3] = eva_en_bal_j["h_2"]  # derivative of eva_en_bal with respect to h61
    jacobian[18, 18] = eva_en_bal_j["m_cold"]  # derivative of eva_en_bal with respect to m41
    jacobian[18, 0] = eva_en_bal_j["h_3"]  # derivative of eva_en_bal with respect to h41
    jacobian[18, 1] = eva_en_bal_j["h_4"]  # derivative of eva_en_bal with respect to h42
    jacobian[19, 19] = exp_en_bal_j["P"]  # derivative of exp_en_bal with respect to P_EXP
    jacobian[19, 17] = exp_en_bal_j["m"]  # derivative of exp_en_bal with respect to m61
    jacobian[19, 10] = exp_en_bal_j["h_1"]  # derivative of exp_en_bal with respect to h64
    jacobian[19, 12] = exp_en_bal_j["h_2"]  # derivative of exp_en_bal with respect to h65
    jacobian[20, 20] = pump_en_bal_j["P"]  # derivative of pump_en_bal with respect to P_PUMP
    jacobian[20, 17] = pump_en_bal_j["m"]  # derivative of pump_en_bal with respect to m61
    jacobian[20, 3] = pump_en_bal_j["h_1"]  # derivative of pump_en_bal with respect to h66
    jacobian[20, 5] = pump_en_bal_j["h_2"]  # derivative of pump_en_bal with respect to h61

    # Convert the numpy array to a pandas DataFrame
    df = pd.DataFrame(jacobian)

    # Save the DataFrame as a CSV file
    df.round(4).to_csv('jacobian_orc.csv', index=False)

    variables -= np.linalg.inv(jacobian).dot(residual)

    if variables[3] < 0:
        variables[3] = -variables[3]

    iter += 1

    print(variables)

print(f"Simulation converged successfully after {iter} iterations.")
# TODO [h41, h42, p42, h61, p61, h62, h66, p66, p63, p64, h64, p65, h65, h63, h51, h52, p52, m61, m41, P_EXP, P_PUMP]
# TODO   0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18    19     20
t61 = PSI("T", "H", variables[3], "P", variables[4], wf)
t62 = PSI("T", "H", variables[5], "P", p62, wf)
t63 = PSI("T", "H", variables[13], "P", variables[8], wf)
t64 = PSI("T", "H", variables[10], "P", variables[9], wf)
t65 = PSI("T", "H", variables[12], "P", variables[11], wf)
t66 = PSI("T", "H", variables[6], "P", variables[7], wf)
t51 = PSI("T", "H", variables[14], "P", p51, fluid_TES)
t52 = PSI("T", "H", variables[15], "P", variables[16], fluid_TES)
t41 = PSI("T", "H", variables[0], "P", p41, fluid_ambient)
t42 = PSI("T", "H", variables[1], "P", variables[2], fluid_ambient)

p61 = variables[4]
p63 = variables[8]
p64 = variables[9]
p65 = variables[11]
p66 = variables[7]
p42 = variables[2]
p52 = variables[16]

h61 = variables[3]
h62 = variables[5]
h63 = variables[13]
h64 = variables[10]
h66 = variables[6]
h65 = variables[12]
h51 = variables[14]
h52 = variables[15]
h41 = variables[0]
h42 = variables[1]

m61 = variables[17]
m41 = variables[18]

s61 = PSI("S", "H", h61, "P", p61, wf)
s62 = PSI("S", "H", h62, "P", p62, wf)
s63 = PSI("S", "H", h63, "P", p62, wf)
s64 = PSI("S", "H", h64, "P", p64, wf)
s65 = PSI("S", "H", h65, "P", p65, wf)
s66 = PSI("S", "H", h66, "P", p66, wf)
s51 = PSI("S", "H", h51, "P", p51, fluid_TES)
s52 = PSI("S", "H", h52, "P", p52, fluid_TES)
s41 = PSI("S", "H", h41, "P", p41, fluid_ambient)
s42 = PSI("S", "H", h42, "P", p42, fluid_ambient)

df = pd.DataFrame(index=[61, 62, 63, 64, 65, 66, 51, 52, 41, 42],
                  columns=["m [kg/s]", "T [˚C]", "h [kJ/kg]", "p [bar]", "s [J/kgK]"])
df.loc[61] = [m61, t61, h61, p61, s61]
df.loc[62] = [m61, t62, h62, p62, s62]
df.loc[63] = [m61, t63, h63, p63, s63]
df.loc[64] = [m61, t64, h64, p64, s64]
df.loc[65] = [m61, t65, h65, p65, s65]
df.loc[66] = [m61, t66, h66, p66, s66]
df.loc[51] = [m51, t51, h51, p51, s51]
df.loc[52] = [m51, t52, h52, p51, s52]
df.loc[41] = [m41, t41, h41, p41, s41]
df.loc[42] = [m41, t42, h42, p42, s42]

df["T [˚C]"] = df["T [˚C]"] - 273.15
df["h [kJ/kg]"] = df["h [kJ/kg]"] * 1e-3
df["p [bar]"] = df["p [bar]"] * 1e-5

print(df)

P_PUMP = df.loc[61, "m [kg/s]"] * (df.loc[62, "h [kJ/kg]"] - df.loc[61, "h [kJ/kg]"])
P_EXP = df.loc[61, "m [kg/s]"] * (df.loc[64, "h [kJ/kg]"] - df.loc[65, "h [kJ/kg]"])
Q_eva = df.loc[51, "m [kg/s]"] * (df.loc[51, "h [kJ/kg]"] - df.loc[52, "h [kJ/kg]"])
Q_cond = df.loc[41, "m [kg/s]"] * (df.loc[42, "h [kJ/kg]"] - df.loc[41, "h [kJ/kg]"])

if round(Q_eva+P_PUMP-P_EXP-Q_cond) > 1e-5:
    print("Energy balances are not fulfilled! :(")
