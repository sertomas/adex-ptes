import numpy as np
from CoolProp.CoolProp import PropsSI
import pandas as pd

def pr_func(pr, p_1, p_2):
    return pr * p_1 - p_2


def pr_deriv(pr, p_1, p_2):
    return {
        "p_1": pr,
        "p_2": -1
    }


def eta_s_func(eta_s, h_1, p_1, h_2, p_2, fluid):
    h_2s = PropsSI("H", "P", p_2, "S", PropsSI("S", "H", h_1, "P", p_1, fluid), fluid)
    return (h_2 - h_1) * eta_s - (h_2s - h_1)


def eta_s_deriv(eta_s, h_1, p_1, h_2, p_2, fluid):
    d = 1e-2
    return {
        "h_1": (eta_s_func(eta_s, h_1 + d, p_1, h_2, p_2, fluid) - eta_s_func(eta_s, h_1 - d, p_1, h_2, p_2, fluid)) / (2 * d),
        "h_2": eta_s,
        "p_1": (eta_s_func(eta_s, h_1, p_1 + d, h_2, p_2, fluid) - eta_s_func(eta_s, h_1, p_1 - d, h_2, p_2, fluid)) / (2 * d),
        "p_2": (eta_s_func(eta_s, h_1, p_1, h_2, p_2 + d, fluid) - eta_s_func(eta_s, h_1, p_1, h_2, p_2 - d, fluid)) / (2 * d),
    }


def turbo_func(P, m, h_1, h_2):
    return - P + m * (h_2 - h_1)


def turbo_deriv(P, m, h_1, h_2):
    return {
        "P": -1,
        "m": (h_2 - h_1),
        "h_1": -m,
        "h_2": m
    }


def he_func(m_hot, h_1, h_2, m_cold, h_3, h_4):
    return m_hot * (h_1 - h_2) + m_cold * (h_3 - h_4)


def he_deriv(m_hot, h_1, h_2, m_cold, h_3, h_4):
    return {
        "m_hot": (h_1 - h_2),
        "m_cold": -(h_3 - h_4),
        "h_1": m_hot,
        "h_2": -m_hot,
        "h_3": m_cold,
        "h_4": -m_cold
    }


def ihx_func(h_1, h_2, h_3, h_4):
    return (h_1 - h_2) + (h_3 - h_4)


def ihx_deriv(h_1, h_2, h_3, h_4):
    return {
        "h_1": 1,
        "h_2": -1,
        "h_3": 1,
        "h_4": -1
    }


def temperature_func(T, h, p, fluid):
    return PropsSI("T", "H", h, "P", p, fluid) - T


def temperature_deriv(T, h, p, fluid):
    d = 1e-2
    return {
        "h": (PropsSI("T", "H", h + d, "P", p, fluid) - PropsSI("T", "H", h - d, "P", p, fluid)) / (2 * d),
        "p": (PropsSI("T", "H", h, "P", p + d, fluid) - PropsSI("T", "H", h, "P", p - d, fluid)) / (2 * d)
    }


def valve_func(h_1, h_2):
    return h_1 - h_2


def valve_deriv(h_1, h_2):
    return {
        "h_1": 1,
        "h_2": -1
    }


def x_saturation_func(Q, h, p, fluid):
    return PropsSI("Q", "H", h, "P", p, fluid) - Q


def x_saturation_deriv(Q, h, p, fluid):
    d = 1e-2
    return {
        "h": (PropsSI("Q", "H", h + d, "P", p, fluid) - PropsSI("Q", "H", h - d, "P", p, fluid)) / (2 * d),
        "p": (PropsSI("Q", "H", h, "P", p + d, fluid) - PropsSI("Q", "H", h, "P", p - d, fluid)) / (2 * d)
    }


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
pr_cond_hot = 0.95
pr_ihx_hot = 0.985
pr_ihx_cold = 0.985
pr_eva_cold = 0.95
pr_eva_hot = 1


# AMBIENT
t11 = 10 + 273.15  # input is known
p11 = 1.013 * 1e5  # input is known
t12 = 7 + 273.15   # output temperature is set

# HEAT PUMP
ttd_l_COND = 5
ttd_u_IHX = 5
ttd_l_EVA = 5
p32 = 12.5 * 1e5  # design variable to optimize

# TECHNICAL PARAMETERS
# eta_s = 0.844949  # --> it seems to be different from TESPy results (look T_31)
eta_s = 0.85

# pre-calculation
t32 = t21 + ttd_l_COND
t34 = t12 - ttd_l_EVA
t36 = t32 - ttd_u_IHX

# STARTING VALUES
h36 = 435e3
h31 = 515e3
p31 = 12e5
h32 = 280e3
m31 = 12.5
power = 1000e3
h21 = 300e3
h22 = 600e3
h33 = 250e3
h35 = 390e3
p36 = 0.26 * 1e5
p33 = 12.3 * 1e5
h34 = 220e3
h11 = 415e3
h12 = 400e3
m11 = 595
p34 = 0.27 * 1e5
p12 = 1 * 1e5
p22 = 5 * 1e5
p35 = 0.2 * 1e5

variables = np.array([h36, h31, p31, h32, m31, power, h21, h22, h33, h35, p36, p33, h34, h11, h12, m11, p34, p12, p22, p35])
residual = np.ones(20)

iter = 0

while np.linalg.norm(residual) > 1e-4:
    # TODO [h36, h31, p31, h32, m31, power, h21, h22, h33, h35, p36, p33, h34, h11, h12, m11, p34, p12, p22, p35])
    # TODO   0    1    2    3    4     5     6    7    8    9    10   11   12   13   14   15   16   17   18   19
    eta_s_def = eta_s_func(eta_s, variables[0], variables[10], variables[1], variables[2], wf)
    t36_set = temperature_func(t36, variables[0], variables[10], wf)
    eva_en_bal = he_func(variables[15], variables[13], variables[14], variables[4], variables[12], variables[9])
    t32_set = temperature_func(t32, variables[3], p32, wf)
    p31_set = pr_func(pr_cond_hot, variables[2], p32)
    turbo_en_bal = turbo_func(variables[5], variables[4], variables[0], variables[1])
    cond_en_bal = he_func(variables[4], variables[1], variables[3], m21, variables[6], variables[7])
    t21_set = temperature_func(t21, variables[6], p21, fluid_TES)
    t22_set = temperature_func(t22, variables[7], p21, fluid_TES)  # TODO correct p22 with pr_cond_cold
    ihx_en_bal = ihx_func(variables[3], variables[8], variables[9], variables[0])
    p36_set = pr_func(pr_ihx_cold, variables[19], variables[10])
    p33_set = pr_func(pr_ihx_hot, p32, variables[11])
    valve_en_bal = valve_func(variables[8], variables[12])
    t11_set = temperature_func(t11, variables[13], p11, fluid_ambient)
    t12_set = temperature_func(t12, variables[14], p12, fluid_ambient)
    eva_outlet_sat = x_saturation_func(1, variables[9], variables[19], wf)
    p35_set = pr_func(pr_eva_cold, variables[16], variables[19])
    p12_set = pr_func(pr_eva_hot, p11, variables[17])
    p22_set = pr_func(pr_cond_cold, p21, variables[18])
    t34_set = temperature_func(t34, variables[12], variables[16], wf)

    residual = np.array([eta_s_def, t36_set, eva_en_bal, t32_set, p31_set, turbo_en_bal, cond_en_bal, t21_set, t22_set,
                         ihx_en_bal, p36_set, p33_set, valve_en_bal, t11_set, t12_set, eva_outlet_sat, p35_set, p12_set,
                         p22_set, t34_set])
    jacobian = np.zeros((20, 20))

    eta_s_def_j = eta_s_deriv(eta_s, variables[0], variables[10], variables[1], variables[2], wf)
    t36_set_j = temperature_deriv(t36, variables[0], variables[10], wf)
    eva_en_bal_j = he_deriv(variables[15], variables[13], variables[14], variables[4], variables[12], variables[9])
    t32_set_j = temperature_deriv(t32, variables[3], p32, wf)
    p31_set_j = pr_deriv(pr_cond_hot, variables[2], p32)
    turbo_en_bal_j = turbo_deriv(variables[5], variables[4], variables[0], variables[1])
    cond_en_bal_j = he_deriv(variables[4], variables[1], variables[3], m21, variables[6], variables[7])
    t21_set_j = temperature_deriv(t21, variables[6], p21, fluid_TES)
    t22_set_j = temperature_deriv(t22, variables[7], p21, fluid_TES)  # TODO correct p22 with pr_cond_cold
    ihx_en_bal_j = ihx_deriv(variables[3], variables[8], variables[9], variables[0])
    p36_set_j = pr_deriv(pr_ihx_cold, variables[19], variables[10])
    p33_set_j = pr_deriv(pr_ihx_hot, p32, variables[11])
    valve_en_bal_j = valve_deriv(variables[8], variables[12])
    t11_set_j = temperature_deriv(t11, variables[13], p11, fluid_ambient)
    t12_set_j = temperature_deriv(t12, variables[14], p12, fluid_ambient)
    eva_outlet_sat_j = x_saturation_deriv(1, variables[9], variables[19], wf)
    p35_set_j = pr_deriv(pr_eva_cold, variables[16], variables[19])
    p12_set_j = pr_deriv(pr_eva_hot, p11, variables[17])
    p22_set_j = pr_deriv(pr_cond_cold, p21, variables[18])
    t34_set_j = temperature_deriv(t34, variables[12], variables[16], wf)

    # TODO [h36, h31, p31, h32, m31, power, h21, h22, h33, h35, p36, p33, h34, h11, h12, m11, p34, p12, p22, p35])
    # TODO   0    1    2    3    4     5     6    7    8    9    10   11   12   13   14   15   16   17   18   19
    jacobian[0, 0] = eta_s_def_j["h_1"]  # derivative of eta_s_def with respect to h36
    jacobian[0, 10] = eta_s_def_j["p_1"]  # derivative of eta_s_def with respect to p36
    jacobian[0, 1] = eta_s_def_j["h_2"]  # derivative of eta_s_def with respect to h31
    jacobian[0, 2] = eta_s_def_j["p_2"]  # derivative of eta_s_def with respect to p31
    jacobian[1, 0] = t36_set_j["h"]  # derivative of t36_set with respect to h36
    jacobian[1, 10] = t36_set_j["p"]  # derivative of t36_set with respect to p36
    jacobian[2, 15] = eva_en_bal_j["m_hot"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 13] = eva_en_bal_j["h_1"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 14] = eva_en_bal_j["h_2"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 4] = eva_en_bal_j["m_cold"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 12] = eva_en_bal_j["h_3"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 9] = eva_en_bal_j["h_4"]  # derivative of eva_en_bal with respect to h35
    jacobian[3, 3] = t32_set_j["h"]  # derivative of t32_set with respect to h32
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
    jacobian[9, 3] = ihx_en_bal_j["h_1"]  # derivative of ihx_en_bal with respect to h32
    jacobian[9, 8] = ihx_en_bal_j["h_2"]  # derivative of ihx_en_bal with respect to h33
    jacobian[9, 9] = ihx_en_bal_j["h_3"]  # derivative of ihx_en_bal with respect to h35
    jacobian[9, 0] = ihx_en_bal_j["h_4"]  # derivative of ihx_en_bal with respect to h36
    jacobian[10, 10] = p36_set_j["p_2"]  # derivative of p36_set with respect to p36
    jacobian[10, 19] = p36_set_j["p_1"]  # derivative of p36_set with respect to p35
    jacobian[11, 11] = p33_set_j["p_2"]  # derivative of p33_set with respect to p33
    jacobian[12, 12] = valve_en_bal_j["h_2"]  # derivative of valve_en_bal with respect to h34
    jacobian[13, 13] = t11_set_j["h"]  # derivative of t11_set with respect to h11
    jacobian[14, 14] = t12_set_j["h"]  # derivative of t12_set with respect to h12
    jacobian[15, 9] = eva_outlet_sat_j["h"]  # derivative of eva_outlet_sat with respect to h35
    jacobian[15, 19] = eva_outlet_sat_j["p"]  # derivative of eva_outlet_sat with respect to p35
    jacobian[16, 16] = p35_set_j["p_1"]  # derivative of p34_set with respect to p34
    jacobian[16, 19] = p35_set_j["p_2"]  # derivative of p34_set with respect to p35
    jacobian[17, 17] = p12_set_j["p_2"]  # derivative of p12_set with respect to p33
    jacobian[18, 18] = p22_set_j["p_2"]  # derivative of p22_set with respect to p22
    jacobian[19, 12] = t34_set_j["h"]  # derivative of t34_set with respect to h34
    jacobian[19, 16] = t34_set_j["p"]  # derivative of t34_set with respect to p34

    # Convert the numpy array to a pandas DataFrame
    df = pd.DataFrame(jacobian)

    # Save the DataFrame as a CSV file
    df.round(4).to_csv('jacobian_matrix.csv', index=False)

    variables -= np.linalg.inv(jacobian).dot(residual)

    iter += 1
    print(iter)
# TODO [h36, h31, p31, h32, m31, power, h21, h22, h33, h35, p36, p33, h34, h11, h12, m11, p34, p12, p22, p35])
# TODO   0    1    2    3    4     5     6    7    8    9    10   11   12   13   14   15   16   17   18   19
t31 = PropsSI("T", "H", variables[1], "P", variables[2], wf)
t32 = PropsSI("T", "H", variables[3], "P", p32, wf)
t33 = PropsSI("T", "H", variables[8], "P", variables[11], wf)
t34 = PropsSI("T", "H", variables[12], "P", variables[16], wf)
t35 = PropsSI("T", "H", variables[9], "P", variables[19], wf)
t36 = PropsSI("T", "H", variables[0], "P", variables[10], wf)
t21 = PropsSI("T", "H", variables[6], "P", p21, fluid_TES)
t22 = PropsSI("T", "H", variables[7], "P", p22, fluid_TES)
t11 = PropsSI("T", "H", variables[13], "P", p11, fluid_ambient)
t12 = PropsSI("T", "H", variables[14], "P", variables[17], fluid_ambient)

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

s31 = PropsSI("S", "H", h31, "P", p31, wf)
s32 = PropsSI("S", "H", h32, "P", p32, wf)
s33 = PropsSI("S", "H", h33, "P", p32, wf)
s34 = PropsSI("S", "H", h34, "P", p34, wf)
s35 = PropsSI("S", "H", h35, "P", p35, wf)
s36 = PropsSI("S", "H", h36, "P", p36, wf)
s21 = PropsSI("S", "H", h21, "P", p21, fluid_TES)
s22 = PropsSI("S", "H", h22, "P", p22, fluid_TES)
s11 = PropsSI("S", "H", h11, "P", p11, fluid_ambient)
s12 = PropsSI("S", "H", h12, "P", p12, fluid_ambient)

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

P_comp = df.loc[31, "m [kg/s]"] * (df.loc[31, "h [kJ/kg]"] - df.loc[36, "h [kJ/kg]"])
Q_eva = df.loc[11, "m [kg/s]"] * (df.loc[11, "h [kJ/kg]"] - df.loc[12, "h [kJ/kg]"])
Q_cond = df.loc[21, "m [kg/s]"] * (df.loc[22, "h [kJ/kg]"] - df.loc[21, "h [kJ/kg]"])

if round(P_comp+Q_eva-Q_cond) > 1e-5:
    print("Energy balances are not fulfilled! :(")