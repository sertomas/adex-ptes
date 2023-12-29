import numpy as np
from CoolProp.CoolProp import PropsSI as PSI
import pandas as pd

from func_fix import (pr_func, pr_deriv, eta_s_PUMP_func, eta_s_PUMP_deriv, turbo_func, turbo_deriv, he_func, he_deriv,
                         ihx_func, ihx_deriv, temperature_func, temperature_deriv, valve_func, valve_deriv,
                         x_saturation_func, x_saturation_deriv)

T_0 = 283.15


def eps_comp_func(epsilon, h_1, p_1, h_2, p_2, fluid):
    s_2 = PSI("S", "H", h_2, "P", p_2, fluid)
    s_1 = PSI("S", "H", h_1, "P", p_1, fluid)
    return (h_2 - h_1) * epsilon - (h_2 - h_1 - T_0 * (s_2 - s_1))


def eps_comp_deriv(epsilon, h_1, p_1, h_2, p_2, fluid):
    d = 1
    return {
        "h_1": (eps_comp_func(epsilon, h_1 + d, p_1, h_2, p_2, fluid) - eps_comp_func(epsilon, h_1 - d, p_1, h_2, p_2, fluid)) / (2 * d),
        "h_2": (eps_comp_func(epsilon, h_1, p_1, h_2 + d, p_2, fluid) - eps_comp_func(epsilon, h_1, p_1, h_2 + d, p_2, fluid)) / (2 * d),
        "p_1": (eps_comp_func(epsilon, h_1, p_1 + d, h_2, p_2, fluid) - eps_comp_func(epsilon, h_1, p_1 - d, h_2, p_2, fluid)) / (2 * d),
        "p_2": (eps_comp_func(epsilon, h_1, p_1, h_2, p_2 + d, fluid) - eps_comp_func(epsilon, h_1, p_1, h_2, p_2 - d, fluid)) / (2 * d),
    }


def eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, 
                  h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold):
    s_hot_in = PSI("S", "H", h_hot_in, "P", p_hot_in, fluid_hot)
    s_hot_out = PSI("S", "H", h_hot_out, "P", p_hot_out, fluid_hot)
    s_cold_in = PSI("S", "H", h_cold_in, "P", p_cold_in, fluid_cold)
    s_cold_out = PSI("S", "H", h_cold_out, "P", p_cold_out, fluid_cold)
    return (m_hot * (h_hot_in - h_hot_out - T_0 * (s_hot_in - s_hot_out)) * epsilon 
            - m_cold * (h_cold_in - h_cold_out - T_0 * (s_cold_in - s_cold_out)))


def eps_cond_deriv(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot,
                   h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold):
    d = 1
    return {
        "h_hot_in": (eps_cond_func(epsilon, h_hot_in + d, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold) 
                     - eps_cond_func(epsilon, h_hot_in - d, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold) / (2 * d)),
        "h_hot_out": (eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out + d, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold) 
                      - eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out - d, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold) / (2 * d)),
        "p_hot_in": (eps_cond_func(epsilon, h_hot_in, p_hot_in + d, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold) 
                     - eps_cond_func(epsilon, h_hot_in, p_hot_in - d, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold) / (2 * d)),
        "p_hot_out": (eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out + d, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold) 
                      - eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out - d, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold) / (2 * d)),
        "h_cold_in": (eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in + d, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold) 
                      - eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in - d, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold) / (2 * d)),
        "h_cold_out": (eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out + d, p_cold_out, m_cold, fluid_cold) 
                       - eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out - d, p_cold_out, m_cold, fluid_cold) / (2 * d)),
        "p_cold_in": (eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in + d, h_cold_out, p_cold_out, m_cold, fluid_cold) 
                      - eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in - d, h_cold_out, p_cold_out, m_cold, fluid_cold) / (2 * d)),
        "p_cold_out": (eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out + d, m_cold, fluid_cold) 
                       - eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out - d, m_cold, fluid_cold) / (2 * d)),
        "m_hot": (eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot + d, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                  - eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot - d, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold) / (2 * d)),
        "m_cold": (eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold + d, fluid_cold)
                   - eps_cond_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold - d, fluid_cold) / (2 * d)),

    }


epsilon = pd.read_csv("outputs/exan/hp_base_exan_components.csv", index_col=0)["epsilon"]

wf = "REFPROP::R1336MZZZ"
fluid_TES = "REFPROP::water"
fluid_ambient = "REFPROP::air"

# TES
t21 = 70 + 273.15    # input is known
p21 = 5 * 1e5        # input is known
t22 = 140 + 273.15   # output temperature is set
m21 = 10             # dimensioning of the system (full load)

# PRESSURE DROPS
pr_cond_cold = 1
pr_cond_hot = 1
pr_ihx_hot = 1
pr_ihx_cold = 1
pr_eva_cold = 1
pr_eva_hot = 1


# AMBIENT
t11 = 10 + 273.15  # input is known
p11 = 1.013 * 1e5  # input is known
t12 = 9.999999 + 273.15   # output temperature is set

# HEAT PUMP
ttd_l_COND = 0
ttd_u_IHX = 0
ttd_l_EVA = 0
p32 = 13.75 * 1e5  # design variable to optimize

# TECHNICAL PARAMETERS
# eta_s = 0.844949  # --> it seems to be different from TESPy results (look T_31)
eta_s = 1

# pre-calculation
t32 = t21 + ttd_l_COND
t34 = t12 - ttd_l_EVA
t36 = t32 - ttd_u_IHX

# STARTING VALUES
h36 = 439e3
h31 = 506e3
p31 = 13.7e5
h32 = 292e3
m31 = 13.5
power = 900e3
h21 = 295e3
h22 = 590e3
h33 = 233e3
h35 = 385e3
p36 = 0.4e5
p33 = 13.7e5
h34 = 230e3
h11 = 410e3
h12 = 409e3
m11 = 220700
p34 = 0.4e5
p12 = 1e5
p22 = 4.9e5
p35 = 0.4e5

variables = np.array([h36, h31, p31, h32, m31, power, h21, h22, p22])
residual = np.ones(9)

iter = 0

while np.linalg.norm(residual) > 1e-4:
    # TODO [h36, h31, p31, h32, m31, power, h21, h22, p22])
    # TODO   0    1    2    3    4     5     6    7    8  
    eps_comp_def = eta_s_PUMP_func(1, variables[0], p36, variables[1], variables[2], wf)
    t36_set = temperature_func(t36, variables[0], p36, wf)
    eps_cond_def = eps_cond_func(1, variables[1], variables[2], variables[3], p32, variables[4], wf, 
                                 variables[6], p21, variables[7], variables[8], m21, fluid_TES)
    p31_set = pr_func(pr_cond_hot, variables[2], p32)
    turbo_en_bal = turbo_func(variables[5], variables[4], variables[0], variables[1])
    cond_en_bal = he_func(variables[4], variables[1], variables[3], m21, variables[6], variables[7])
    t21_set = temperature_func(t21, variables[6], p21, fluid_TES)
    t22_set = temperature_func(t22, variables[7], p21, fluid_TES)  # TODO correct p22 with pr_cond_cold
    p22_set = pr_func(pr_cond_cold, p21, variables[8])

    residual = np.array([eps_comp_def, t36_set, eps_cond_def, p31_set, turbo_en_bal, cond_en_bal, t21_set, t22_set, p22_set])
    jacobian = np.zeros((9, 9))

    eps_comp_def_j = eta_s_PUMP_deriv(1, variables[0], p36, variables[1], variables[2], wf)
    t36_set_j = temperature_deriv(t36, variables[0], p36, wf)
    eps_cond_def_j = eps_cond_deriv(1, variables[1], variables[2], variables[3], p32, variables[4], wf,
                                    variables[6], p21, variables[7], variables[8], m21, fluid_TES)
    p31_set_j = pr_deriv(pr_cond_hot, variables[2], p32)
    turbo_en_bal_j = turbo_deriv(variables[5], variables[4], variables[0], variables[1])
    cond_en_bal_j = he_deriv(variables[4], variables[1], variables[3], m21, variables[6], variables[7])
    t21_set_j = temperature_deriv(t21, variables[6], p21, fluid_TES)
    t22_set_j = temperature_deriv(t22, variables[7], p21, fluid_TES)  # TODO correct p22 with pr_cond_cold
    p22_set_j = pr_deriv(pr_cond_cold, p21, variables[8])

# def eps_cond_deriv(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot,
    #                    h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold):

    # TODO [h36, h31, p31, h32, m31, power, h21, h22, p22])
    # TODO   0    1    2    3    4     5     6    7    8
    jacobian[0, 0] = eps_comp_def["h_1"]  # derivative of eta_s_def with respect to h36
    jacobian[0, 1] = eps_comp_def["h_2"]  # derivative of eta_s_def with respect to h31
    jacobian[0, 2] = eps_comp_def["p_2"]  # derivative of eta_s_def with respect to p31
    jacobian[1, 0] = t36_set_j["h"]  # derivative of t36_set with respect to h36
    jacobian[2, 1] = eps_cond_def_j["h_hot_in"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 2] = eps_cond_def_j["p_hot_in"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 3] = eps_cond_def_j["h_hot_out"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 4] = eps_cond_def_j["m_hot"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 6] = eps_cond_def_j["h_cold_in"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 7] = eps_cond_def_j["p_cold"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 8] = eps_comp_def_j["h"]  # derivative of t32_set with respect to h32
    jacobian[4, 2] = p31_set_j["p_1"]  # derivative of p31_set with respect to p31
    jacobian[5, 5] = turbo_en_bal_j["P"]  # derivative of turbo_en_bal with respect to power
    jacobian[5, 4] = turbo_en_bal_j["m"]  # derivative of turbo_en_bal with respect to m31
    jacobian[5, 0] = turbo_en_bal_j["h_1"]  # derivative of turbo_en_bal with respect to h36
    jacobian[5, 1] = turbo_en_bal_j["h_2"]  # derivative of turbo_en_bal with respect to h31
    jacobian[6, 1] = cond_en_bal_j["h_1"]  # derivative of cond_en_bal with respect to h31
    jacobian[6, 3] = cond_en_bal_j["h_2"]  # derivative of cond_en_bal with respect to h32
    jacobian[6, 4] = cond_en_bal_j["m_hot"]  # derivative of cond_en_bal with respect to m31
    jacobian[6, 6] = cond_en_bal_j["h_3"]  # derivative of cond_en_bal with respect to h21
    jacobian[6, 7] = cond_en_bal_j["h_4"]  # derivative of cond_en_bal with respect to h22
    jacobian[7, 6] = t21_set_j["h"]  # derivative of t21_set with respect to h21
    jacobian[8, 7] = t22_set_j["h"]  # derivative of t22_set with respect to h22

    # Convert the numpy array to a pandas DataFrame
    df = pd.DataFrame(jacobian)

    # Save the DataFrame as a CSV file
    df.round(4).to_csv('jacobian_matrix.csv', index=False)

    variables -= np.linalg.inv(jacobian).dot(residual)

    # if variables[15] < 0:
    #    variables[15] = -variables[15]

    iter += 1
    print(iter)
# TODO [h36, h31, p31, h32, m31, power, h21, h22, h33, h35, p36, p33, h34, h11, h12, m11, p34, p12, p22, p35])
# TODO   0    1    2    3    4     5     6    7    8    9    10   11   12   13   14   15   16   17   18   19
t31 = PSI("T", "H", variables[1], "P", variables[2], wf)
t32 = PSI("T", "H", variables[3], "P", p32, wf)
t33 = PSI("T", "H", variables[8], "P", variables[11], wf)
t34 = PSI("T", "H", variables[12], "P", variables[16], wf)
t35 = PSI("T", "H", variables[9], "P", variables[19], wf)
t36 = PSI("T", "H", variables[0], "P", variables[10], wf)
t21 = PSI("T", "H", variables[6], "P", p21, fluid_TES)
t22 = PSI("T", "H", variables[7], "P", p22, fluid_TES)
t11 = PSI("T", "H", variables[13], "P", p11, fluid_ambient)
t12 = PSI("T", "H", variables[14], "P", variables[17], fluid_ambient)

p31 = variables[2]
p33 = variables[11]
p34 = variables[16]
p35 = variables[19]
p36 = variables[10]
p12 = variables[17]
p22 = variables[18]

h31 = variables[1]
h32 = variables[3]
h33 = variables[8]
h34 = variables[12]
h36 = variables[0]
h35 = variables[9]
h21 = variables[6]
h22 = variables[7]
h11 = variables[13]
h12 = variables[14]

m31 = variables[4]
m11 = variables[15]

s31 = PSI("S", "H", h31, "P", p31, wf)
s32 = PSI("S", "H", h32, "P", p32, wf)
s33 = PSI("S", "H", h33, "P", p32, wf)
s34 = PSI("S", "H", h34, "P", p34, wf)
s35 = PSI("S", "H", h35, "P", p35, wf)
s36 = PSI("S", "H", h36, "P", p36, wf)
s21 = PSI("S", "H", h21, "P", p21, fluid_TES)
s22 = PSI("S", "H", h22, "P", p22, fluid_TES)
s11 = PSI("S", "H", h11, "P", p11, fluid_ambient)
s12 = PSI("S", "H", h12, "P", p12, fluid_ambient)

df = pd.DataFrame(index=[31, 32, 33, 34, 35, 36, 21, 22, 11, 12],
                  columns=["m [kg/s]", "T [˚C]", "h [kJ/kg]", "p [bar]", "s [J/kgK]"])
df.loc[31] = [m31, t31, h31, p31, s31]
df.loc[32] = [m31, t32, h32, p32, s32]
df.loc[33] = [m31, t33, h33, p33, s33]
df.loc[34] = [m31, t34, h34, p34, s34]
df.loc[35] = [m31, t35, h35, p35, s35]
df.loc[36] = [m31, t36, h36, p36, s36]
df.loc[21] = [m21, t21, h21, p21, s21]
df.loc[22] = [m21, t22, h22, p21, s22]
df.loc[11] = [m11, t11, h11, p11, s11]
df.loc[12] = [m11, t12, h12, p12, s12]

df["T [˚C]"] = df["T [˚C]"] - 273.15
df["h [kJ/kg]"] = df["h [kJ/kg]"] * 1e-3
df["p [bar]"] = df["p [bar]"] * 1e-5

print(df)
