import numpy as np
from CoolProp.CoolProp import PropsSI as PSI
import pandas as pd

from func_fix import (pr_func, pr_deriv, eta_s_PUMP_func, eta_s_PUMP_deriv, turbo_func, turbo_deriv, he_func, he_deriv,
                      ihx_func, ihx_deriv, temperature_func, temperature_deriv, valve_func, valve_deriv,
                      x_saturation_func, x_saturation_deriv, eps_compressor_func, eps_compressor_deriv, eps_he_func,
                      eps_he_deriv, eps_ihx_func, eps_ihx_deriv)

T_0 = 283.15

epsilon = pd.read_csv("hp_simult_epsilon.csv", index_col=0)["Value"]
print("epsilon_IHX = ", round(epsilon["IHX"], 5))
print("epsilon_COND = ", round(epsilon["COND"], 5))
print("epsilon_COMP = ", round(epsilon["COMP"], 5))

wf = "REFPROP::R1336MZZZ"
fluid_TES = "REFPROP::water"
fluid_ambient = "REFPROP::air"

# TES
t21 = 70 + 273.15  # input is known
p21 = 5e5  # input is known
t22 = 140 + 273.15  # output temperature is set
m21 = 10  # dimensioning of the system (full load)

# PRESSURE DROPS
pr_cond_cold = 1
pr_cond_hot = 0.95
pr_ihx_hot = 0.985
pr_ihx_cold = 0.985
pr_eva_cold = 0.95

pr_cond_part_cold = np.cbrt(pr_cond_cold)  # COND pressure drop is split equally (geom, mean) between ECO-EVA-SH
pr_cond_part_hot = np.cbrt(pr_cond_hot)  # COND pressure drop is split equally (geom, mean) between ECO-EVA-SH

# AMBIENT
t0 = 25 + 273.15  # K
p0 = 1.013e5  # bar

# HEAT PUMP
ttd_l_cond = 5  # K
ttd_u_ihx = 5  # K
ttd_l_eva = 5  # K
target_p32 = 14.501436 * 1e5  # from optimal base case

# TECHNICAL PARAMETERS
eta_s = 0.85

# pre-calculation
t32 = t21 + ttd_l_cond
t34 = t0 - ttd_l_eva
t36 = t32 - ttd_u_ihx

# STARTING VALUES
h31 = 525.480e3
h32 = 293.761e3
h33 = 233.804e3
h34 = 233.803e3
h35 = 380.150e3
h36 = 440.108e3
p31 = 12e5
p33 = 12.8 * 1e5
p34 = 0.272 * 1e5
p35 = 0.259 * 1e5
p36 = 0.255 * 1e5
m31 = 12.76
power = 1000e3
h21 = 293e3
h22 = 589e3
h11 = 409e3
h12 = 406e3
p12 = 1 * 1e5
p22 = 5 * 1e5
m11 = 615

variables = np.array([h36, h31, p31, h32, m31, power, h21, h22, h33, h35, p36, p33, h34, h11, h12, m11, p34, p12, p22, p35])
residual = np.ones(20)

iter = 0

while np.linalg.norm(residual) > 1e-4:
    # TODO [h36, h31, p31, h32, m31, power, h21, h22, h33, h35, p36, p33, h34, h11, h12, m11, p34, p12, p22, p35]
    # TODO   0    1    2    3    4     5     6    7    8    9    10   11   12   13   14   15   16   17   18   19
    # eta_s_def = eta_s_PUMP_func(eta_s, variables[0], variables[10], variables[1], variables[2], wf)
    eta_s_def = eps_compressor_func(epsilon['COMP'], variables[0], variables[10], variables[1], variables[2], wf)
    # t36_set = temperature_func(t36, variables[0], variables[10], wf)
    eps_ihx_def = eps_ihx_func(epsilon['IHX'], variables[3], target_p32, variables[8], variables[11], wf,
                               variables[9], variables[19], variables[0], variables[10], wf)
    eva_en_bal = he_func(variables[15], variables[13], variables[14], variables[4], variables[12], variables[9])
    t32_set = temperature_func(t32, variables[3], target_p32, wf)
    # eps_cond_def = eps_he_func(epsilon['COND'], variables[1], variables[2], variables[3], target_p32, variables[4], wf,
                               # variables[6], p21, variables[7], variables[18], m21, fluid_TES)
    p31_set = pr_func(pr_cond_hot, variables[2], target_p32)
    turbo_en_bal = turbo_func(variables[5], variables[4], variables[0], variables[1])
    cond_en_bal = he_func(variables[4], variables[1], variables[3], m21, variables[6], variables[7])
    t21_set = temperature_func(t21, variables[6], p21, fluid_TES)
    t22_set = temperature_func(t22, variables[7], p22, fluid_TES)  # TODO correct p22 with pr_cond_cold
    ihx_en_bal = ihx_func(variables[3], variables[8], variables[9], variables[0])
    p36_set = pr_func(pr_ihx_cold, variables[19], variables[10])
    p33_set = pr_func(pr_ihx_hot, target_p32, variables[11])
    valve_en_bal = valve_func(variables[8], variables[12])
    t11_set = temperature_func(t11, variables[13], p11, fluid_ambient)
    t12_set = temperature_func(t12, variables[14], p12, fluid_ambient)
    eva_outlet_sat = x_saturation_func(1, variables[9], variables[19], wf)
    p35_set = pr_func(pr_eva_cold, variables[16], variables[19])
    p12_set = pr_func(pr_eva_hot, p11, variables[17])
    p22_set = pr_func(pr_cond_cold, p21, variables[18])
    t34_set = temperature_func(t34, variables[12], variables[16], wf)

    residual = np.array([eta_s_def, eps_ihx_def, eva_en_bal, t32_set, p31_set, turbo_en_bal, cond_en_bal, t21_set, t22_set,
                         ihx_en_bal, p36_set, p33_set, valve_en_bal, t11_set, t12_set, eva_outlet_sat, p35_set, p12_set,
                         p22_set, t34_set], dtype=float)
    jacobian = np.zeros((20, 20))

    # eta_s_def_j = eta_s_PUMP_deriv(eta_s, variables[0], variables[10], variables[1], variables[2], wf)
    eta_s_def_j = eps_compressor_deriv(epsilon['COMP'], variables[0], variables[10], variables[1], variables[2], wf)
    # t36_set_j = temperature_deriv(t36, variables[0], variables[10], wf)
    eps_ihx_def_j = eps_ihx_deriv(epsilon['IHX'], variables[3], target_p32, variables[8], variables[11], wf,
                                  variables[9], variables[19], variables[0], variables[10], wf)
    eva_en_bal_j = he_deriv(variables[15], variables[13], variables[14], variables[4], variables[12], variables[9])
    t32_set_j = temperature_deriv(t32, variables[3], target_p32, wf)
    # eps_cond_def_j = eps_he_deriv(epsilon['COND'], variables[1], variables[2], variables[3], target_p32, variables[4], wf,
                                  # variables[6], p21, variables[7], variables[18], m21, fluid_TES)
    p31_set_j = pr_deriv(pr_cond_hot, variables[2], target_p32)
    turbo_en_bal_j = turbo_deriv(variables[5], variables[4], variables[0], variables[1])
    cond_en_bal_j = he_deriv(variables[4], variables[1], variables[3], m21, variables[6], variables[7])
    t21_set_j = temperature_deriv(t21, variables[6], p21, fluid_TES)
    t22_set_j = temperature_deriv(t22, variables[7], p22, fluid_TES)  # TODO correct p22 with pr_cond_cold
    ihx_en_bal_j = ihx_deriv(variables[3], variables[8], variables[9], variables[0])
    p36_set_j = pr_deriv(pr_ihx_cold, variables[19], variables[10])
    p33_set_j = pr_deriv(pr_ihx_hot, target_p32, variables[11])
    valve_en_bal_j = valve_deriv(variables[8], variables[12])
    t11_set_j = temperature_deriv(t11, variables[13], p11, fluid_ambient)
    t12_set_j = temperature_deriv(t12, variables[14], p12, fluid_ambient)
    eva_outlet_sat_j = x_saturation_deriv(1, variables[9], variables[19], wf)
    p35_set_j = pr_deriv(pr_eva_cold, variables[16], variables[19])
    p12_set_j = pr_deriv(pr_eva_hot, p11, variables[17])
    p22_set_j = pr_deriv(pr_cond_cold, p21, variables[18])
    t34_set_j = temperature_deriv(t34, variables[12], variables[16], wf)

    # TODO [h36, h31, p31, h32, m31, power, h21, h22, h33, h35, p36, p33, h34, h11, h12, m11, p34, p12, p22, p35]
    # TODO   0    1    2    3    4     5     6    7    8    9    10   11   12   13   14   15   16   17   18   19
    jacobian[0, 0] = eta_s_def_j["h_1"]  # derivative of eta_s_def with respect to h36
    jacobian[0, 10] = eta_s_def_j["p_1"]  # derivative of eta_s_def with respect to p36
    jacobian[0, 1] = eta_s_def_j["h_2"]  # derivative of eta_s_def with respect to h31
    jacobian[0, 2] = eta_s_def_j["p_2"]  # derivative of eta_s_def with respect to p31
    # This doesn't work correctly because the IXH needs to be dimensioned, otherwise it converges with t35 = t36
    jacobian[1, 3] = eps_ihx_def_j["h_hot_in"]
    jacobian[1, 8] = eps_ihx_def_j["h_hot_out"]
    jacobian[1, 11] = eps_ihx_def_j["p_hot_out"]
    jacobian[1, 9] = eps_ihx_def_j["h_cold_in"]
    jacobian[1, 19] = eps_ihx_def_j["h_cold_out"]
    jacobian[1, 0] = eps_ihx_def_j["p_cold_in"]
    jacobian[1, 10] = eps_ihx_def_j["p_cold_out"]
    # jacobian[1, 0] = t36_set_j["h"]  # derivative of t36_set with respect to h36
    # jacobian[1, 10] = t36_set_j["p"]  # derivative of t36_set with respect to p36
    jacobian[2, 15] = eva_en_bal_j["m_hot"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 13] = eva_en_bal_j["h_1"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 14] = eva_en_bal_j["h_2"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 4] = eva_en_bal_j["m_cold"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 12] = eva_en_bal_j["h_3"]  # derivative of eva_en_bal with respect to h35
    jacobian[2, 9] = eva_en_bal_j["h_4"]  # derivative of eva_en_bal with respect to h35
    jacobian[3, 3] = t32_set_j["h"]  # derivative of t32_set with respect to h32
    '''jacobian[3, 4] = eps_cond_def_j["m_hot"]
    jacobian[3, 1] = eps_cond_def_j["h_hot_in"]
    jacobian[3, 2] = eps_cond_def_j["p_hot_in"]
    jacobian[3, 3] = eps_cond_def_j["h_hot_out"]
    jacobian[3, 6] = eps_cond_def_j["h_cold_in"]
    jacobian[3, 7] = eps_cond_def_j["h_cold_out"]
    jacobian[3, 18] = eps_cond_def_j["p_cold_out"]'''
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

    # if variables[15] < 0:
    #    variables[15] = -variables[15]

    iter += 1

    print("residuals:", np.linalg.norm(residual))

    cond_number = np.linalg.cond(jacobian)
    # print("Condition number:", cond_number)

print(f"Simulation converged successfully after {iter} iterations.")

# TODO [h36, h31, p31, h32, m31, power, h21, h22, h33, h35, p36, p33, h34, h11, h12, m11, p34, p12, p22, p35])
# TODO   0    1    2    3    4     5     6    7    8    9    10   11   12   13   14   15   16   17   18   19
t31 = PSI("T", "H", variables[1], "P", variables[2], wf)
t32 = PSI("T", "H", variables[3], "P", target_p32, wf)
t33 = PSI("T", "H", variables[8], "P", variables[11], wf)
t34 = PSI("T", "H", variables[12], "P", variables[16], wf)
t35 = PSI("T", "H", variables[9], "P", variables[19], wf)
t36 = PSI("T", "H", variables[0], "P", variables[10], wf)
t21 = PSI("T", "H", variables[6], "P", p21, fluid_TES)
t22 = PSI("T", "H", variables[7], "P", p22, fluid_TES)
t11 = PSI("T", "H", variables[13], "P", p11, fluid_ambient)
t12 = PSI("T", "H", variables[14], "P", variables[17], fluid_ambient)
0
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
s32 = PSI("S", "H", h32, "P", target_p32, wf)
s33 = PSI("S", "H", h33, "P", p33, wf)
s34 = PSI("S", "H", h34, "P", p34, wf)
s35 = PSI("S", "H", h35, "P", p35, wf)
s36 = PSI("S", "H", h36, "P", p36, wf)
s21 = PSI("S", "H", h21, "P", p21, fluid_TES)
s22 = PSI("S", "H", h22, "P", p22, fluid_TES)
s11 = PSI("S", "H", h11, "P", p11, fluid_ambient)
s12 = PSI("S", "H", h12, "P", p12, fluid_ambient)

df = pd.DataFrame(index=[31, 32, 33, 34, 35, 36, 21, 22, 11, 12],
                  columns=["m [kg/s]", "T [°C]", "h [kJ/kg]", "p [bar]", "s [J/kgK]"])
df.loc[31] = [m31, t31, h31, p31, s31]
df.loc[32] = [m31, t32, h32, target_p32, s32]
df.loc[33] = [m31, t33, h33, p33, s33]
df.loc[34] = [m31, t34, h34, p34, s34]
df.loc[35] = [m31, t35, h35, p35, s35]
df.loc[36] = [m31, t36, h36, p36, s36]
df.loc[21] = [m21, t21, h21, p21, s21]
df.loc[22] = [m21, t22, h22, p21, s22]
df.loc[11] = [m11, t11, h11, p11, s11]
df.loc[12] = [m11, t12, h12, p12, s12]

df["T [°C]"] = df["T [°C]"] - 273.15
df["h [kJ/kg]"] = df["h [kJ/kg]"] * 1e-3
df["p [bar]"] = df["p [bar]"] * 1e-5

print(df)

h11A = PSI("H", "T", T_0, "P", p11, fluid_ambient)
s11A = PSI("S", "T", T_0, "P", p11, fluid_ambient)
h12A = PSI("H", "T", T_0, "P", p12, fluid_ambient)
s12A = PSI("S", "T", T_0, "P", p12, fluid_ambient)
h0 = PSI("H", "T", T_0, "P", 1.013e5, fluid_ambient)
s0 = PSI("S", "T", T_0, "P", 1.013e5, fluid_ambient)

e12T = h12-h12A - T_0*(s12-s12A)
e11T = h11-h11A - T_0*(s11-s11A)
e11M = h11A-h0 - T_0*(s11A-s0)
e12M = h12A-h0 - T_0*(s12A-s0)

h0 = PSI("H", "T", T_0, "P", 1.013e5, wf)
s0 = PSI("S", "T", T_0, "P", 1.013e5, wf)
e31 = h31-h0 - T_0*(s31-s0)
e32 = h32-h0 - T_0*(s32-s0)
e33 = h33-h0 - T_0*(s33-s0)
e34 = h34-h0 - T_0*(s34-s0)
e35 = h35-h0 - T_0*(s35-s0)
e36 = h36-h0 - T_0*(s36-s0)

h0 = PSI("H", "T", T_0, "P", 1.013e5, fluid_TES)
s0 = PSI("S", "T", T_0, "P", 1.013e5, fluid_TES)
e21 = h21-h0 - T_0*(s21-s0)
e22 = h22-h0 - T_0*(s22-s0)

epsilon_EVA_neu = m11*(e12T-e11T) / (m31*(e34-e35)+m11*(e11M-e12M))
print("epsilon_EVA =", round(epsilon_EVA_neu, 5))

epsilon_IHX_neu = (e36-e35) / (e32-e33)
print("epsilon_IHX =", round(epsilon_IHX_neu, 5))

epsilon_COND_neu = (m21*(e22-e21)) / (m31*(e31-e32))
print("epsilon_COND =", round(epsilon_COND_neu, 5))

epsilon_COMP_neu = (e31-e36) / (h31-h36)
print("epsilon_COMP =", round(epsilon_COMP_neu, 5))

print("EF_IHX =", round((e32-e33), 3))
print("EP_IHX =", round((e36-e35), 3))
'''
for t32 in range(70, 85):
    h32 = PSI("H", "T", t32+273.15, "P", target_p32, wf)
    h0 = PSI("H", "T", T_0, "P", 1.013e5, wf)
    s0 = PSI("S", "T", T_0, "P", 1.013e5, wf)
    e31 = h31 - h0 - T_0 * (s31 - s0)
    e32 = h32 - h0 - T_0 * (s32 - s0)
    m31 = m21 * (h22 - h21) / (h31 - h32)
    epsilon_COND_neu = (m21 * (e22 - e21)) / (m31 * (e31 - e32))
    print(f"m31 = {round(m31, 2)} kg/s")
    print("EF = ", round((m31 * (e31 - e32) / 1000), 2))
    print(f"epsilon_COND at t32={t32}°C:", round(epsilon_COND_neu, 4))
'''