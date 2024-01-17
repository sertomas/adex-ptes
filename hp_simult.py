import numpy as np
from CoolProp.CoolProp import PropsSI as PSI
from scipy.optimize import minimize
import pandas as pd
from func_fix import (pr_func, pr_deriv, eta_s_PUMP_func, eta_s_PUMP_deriv, turbo_func, turbo_deriv, he_func, he_deriv,
                         ihx_func, ihx_deriv, temperature_func, temperature_deriv, valve_func, valve_deriv,
                         x_saturation_func, x_saturation_deriv, qt_diagram)


def hp_simultaneous(target_p32, print_results, plot, case, delta_t_min):

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

    # TECHNICAL PARAMETERS
    # eta_s = 0.844949  # --> it seems to be different from TESPy results (look T_31)
    eta_s = 0.85

    # pre-calculation
    t32 = t21 + ttd_l_COND
    t34 = t12 - ttd_l_EVA
    t36 = t32 - ttd_u_IHX

    # STARTING VALUES
    h36 = 440e3
    h31 = 525e3
    p31 = 12e5
    h32 = 293e3
    m31 = 12.7
    power = 1000
    h21 = 293e3
    h22 = 589e3
    h33 = 233e3
    h35 = 380e3
    p36 = 0.26 * 1e5
    p33 = 12.3 * 1e5
    h34 = 233e3
    h11 = 409e3
    h12 = 406e3
    m11 = 595
    p34 = 0.27 * 1e5
    p12 = 1 * 1e5
    p22 = 5 * 1e5
    p35 = 0.2 * 1e5

    variables = np.array([h36, h31, p31, h32, m31, power, h21, h22, h33, h35, p36, p33, h34, h11, h12, m11, p34, p12, p22, p35])
    residual = np.ones(20)

    iter = 0

    while np.linalg.norm(residual) > 1e-4:
        # TODO [h36, h31, p31, h32, m31, power, h21, h22, h33, h35, p36, p33, h34, h11, h12, m11, p34, p12, p22, p35]
        # TODO   0    1    2    3    4     5     6    7    8    9    10   11   12   13   14   15   16   17   18   19

        #   0
        eta_s_def = eta_s_PUMP_func(eta_s, variables[0], variables[10], variables[1], variables[2], wf)
        #   1
        t36_set = temperature_func(t36, variables[0], variables[10], wf)
        #   2
        t32_set = temperature_func(t32, variables[3], target_p32, wf)
        #   3
        eva_en_bal = he_func(variables[15], variables[13], variables[14], variables[4], variables[12], variables[9])
        #   4
        p31_set = pr_func(pr_cond_hot, variables[2], target_p32)
        #   5
        turbo_en_bal = turbo_func(variables[5], variables[4], variables[0], variables[1])
        #   6
        cond_en_bal = he_func(variables[4], variables[1], variables[3], m21, variables[6], variables[7])
        #   7
        t21_set = temperature_func(t21, variables[6], p21, fluid_TES)
        #   8
        t22_set = temperature_func(t22, variables[7], p22, fluid_TES)  # TODO correct p22 with pr_cond_cold
        #   9
        ihx_en_bal = ihx_func(variables[3], variables[8], variables[9], variables[0])
        #   10
        p36_set = pr_func(pr_ihx_cold, variables[19], variables[10])
        #   11
        p33_set = pr_func(pr_ihx_hot, target_p32, variables[11])
        #   12
        valve_en_bal = valve_func(variables[8], variables[12])
        #   13
        t11_set = temperature_func(t11, variables[13], p11, fluid_ambient)
        #   14
        t12_set = temperature_func(t12, variables[14], p12, fluid_ambient)
        #   15
        eva_outlet_sat = x_saturation_func(1, variables[9], variables[19], wf)
        #   16
        t34_set = temperature_func(t34, variables[12], variables[16], wf)
        #   17
        p12_set = pr_func(pr_eva_hot, p11, variables[17])
        #   18
        p22_set = pr_func(pr_cond_cold, p21, variables[18])
        #   19
        p35_set = pr_func(pr_eva_cold, variables[16], variables[19])

        residual = np.array([eta_s_def, t36_set, t32_set, eva_en_bal, p31_set, turbo_en_bal, cond_en_bal, t21_set, t22_set,
                             ihx_en_bal, p36_set, p33_set, valve_en_bal, t11_set, t12_set, eva_outlet_sat, t34_set, p12_set,
                             p22_set, p35_set], dtype=float)
        jacobian = np.zeros((20, 20))

        eta_s_def_j = eta_s_PUMP_deriv(eta_s, variables[0], variables[10], variables[1], variables[2], wf)
        t36_set_j = temperature_deriv(t36, variables[0], variables[10], wf)
        t32_set_j = temperature_deriv(t32, variables[3], target_p32, wf)
        eva_en_bal_j = he_deriv(variables[15], variables[13], variables[14], variables[4], variables[12], variables[9])
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
        t34_set_j = temperature_deriv(t34, variables[12], variables[16], wf)
        p12_set_j = pr_deriv(pr_eva_hot, p11, variables[17])
        p22_set_j = pr_deriv(pr_cond_cold, p21, variables[18])
        p35_set_j = pr_deriv(pr_eva_cold, variables[16], variables[19])

        # TODO [h36, h31, p31, h32, m31, power, h21, h22, h33, h35, p36, p33, h34, h11, h12, m11, p34, p12, p22, p35]
        # TODO   0    1    2    3    4     5     6    7    8    9    10   11   12   13   14   15   16   17   18   19
        jacobian[0, 0] = eta_s_def_j["h_1"]  # derivative of eta_s_def with respect to h36
        jacobian[0, 10] = eta_s_def_j["p_1"]  # derivative of eta_s_def with respect to p36
        jacobian[0, 1] = eta_s_def_j["h_2"]  # derivative of eta_s_def with respect to h31
        jacobian[0, 2] = eta_s_def_j["p_2"]  # derivative of eta_s_def with respect to p31
        jacobian[1, 0] = t36_set_j["h"]  # derivative of t36_set with respect to h36
        jacobian[1, 10] = t36_set_j["p"]  # derivative of t36_set with respect to p36
        jacobian[2, 3] = t32_set_j["h"]  # derivative of t32_set with respect to h32
        jacobian[3, 15] = eva_en_bal_j["m_hot"]  # derivative of eva_en_bal with respect to h35
        jacobian[3, 13] = eva_en_bal_j["h_1"]  # derivative of eva_en_bal with respect to h35
        jacobian[3, 14] = eva_en_bal_j["h_2"]  # derivative of eva_en_bal with respect to h35
        jacobian[3, 4] = eva_en_bal_j["m_cold"]  # derivative of eva_en_bal with respect to h35
        jacobian[3, 12] = eva_en_bal_j["h_3"]  # derivative of eva_en_bal with respect to h35
        jacobian[3, 9] = eva_en_bal_j["h_4"]  # derivative of eva_en_bal with respect to h35
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
        jacobian[16, 12] = t34_set_j["h"]  # derivative of t34_set with respect to h34
        jacobian[16, 16] = t34_set_j["p"]  # derivative of t34_set with respect to p34
        jacobian[17, 17] = p12_set_j["p_2"]  # derivative of p12_set with respect to p33
        jacobian[18, 18] = p22_set_j["p_2"]  # derivative of p22_set with respect to p22
        jacobian[19, 16] = p35_set_j["p_1"]  # derivative of p34_set with respect to p34
        jacobian[19, 19] = p35_set_j["p_2"]  # derivative of p34_set with respect to p35

        # Convert the numpy array to a pandas DataFrame
        df = pd.DataFrame(jacobian)

        # Save the DataFrame as a CSV file
        df.round(4).to_csv('jacobian_hp.csv', index=False)

        dump = 0.799

        variables -= dump * np.linalg.inv(jacobian).dot(residual)

        iter += 1

        cond_number = np.linalg.cond(jacobian)
        print("Condition number: ", cond_number, " and residual: ", np.linalg.norm(residual))

    print(f"Simulation converged successfully after {iter} iterations.")
    print(f"p32 = {round(target_p32, 3)} bar.")
    # TODO [h36, h31, p31, h32, m31, power, h21, h22, h33, h35, p36, p33, h34, h11, h12, m11, p34, p12, p22, p35]
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
                      columns=["m [kg/s]", "T [°C]", "h [kJ/kg]", "p [bar]", "s [J/kgK]", "fluid"])
    df.loc[31] = [m31, t31, h31, p31, s31, wf]
    df.loc[32] = [m31, t32, h32, target_p32, s32, wf]
    df.loc[33] = [m31, t33, h33, p33, s33, wf]
    df.loc[34] = [m31, t34, h34, p34, s34, wf]
    df.loc[35] = [m31, t35, h35, p35, s35, wf]
    df.loc[36] = [m31, t36, h36, p36, s36, wf]
    df.loc[21] = [m21, t21, h21, p21, s21, fluid_TES]
    df.loc[22] = [m21, t22, h22, p21, s22, fluid_TES]
    df.loc[11] = [m11, t11, h11, p11, s11, fluid_ambient]
    df.loc[12] = [m11, t12, h12, p12, s12, fluid_ambient]

    df["T [°C]"] = df["T [°C]"] - 273.15
    df["h [kJ/kg]"] = df["h [kJ/kg]"] * 1e-3
    df["p [bar]"] = df["p [bar]"] * 1e-5

    if print_results:
        print(df)

    P_comp = df.loc[31, "m [kg/s]"] * (df.loc[31, "h [kJ/kg]"] - df.loc[36, "h [kJ/kg]"])
    Q_eva = df.loc[11, "m [kg/s]"] * (df.loc[11, "h [kJ/kg]"] - df.loc[12, "h [kJ/kg]"])
    Q_cond = df.loc[21, "m [kg/s]"] * (df.loc[22, "h [kJ/kg]"] - df.loc[21, "h [kJ/kg]"])

    if round(P_comp+Q_eva-Q_cond) > 1e-5:
        print("Energy balances are not fulfilled! :(")

    [min_td_cond, max_td_cond] = qt_diagram(df, 'COND', 31, 32, 21, 22, delta_t_min, 'HP',
               plot=plot, case=f'{case}')
    qt_diagram(df, 'EVA', 11, 12, 34, 35, delta_t_min, 'HP',
               plot=plot, case=f'{case}')
    qt_diagram(df, 'IHX', 32, 33, 35, 36, delta_t_min, 'HP',
               plot=plot, case=f'{case}')

    print('P = ', m31 * (h31-h36))

    return df, min_td_cond


def epsilon_func(T_0, p_0, df):
    h11A = PSI("H", "T", T_0, "P", df.loc[11, 'p [bar]'] * 1e5, df.loc[11, 'fluid']) * 1e-3
    s11A = PSI("S", "T", T_0, "P", df.loc[11, 'p [bar]'] * 1e5, df.loc[11, 'fluid'])
    h12A = PSI("H", "T", T_0, "P", df.loc[12, 'p [bar]'] * 1e5, df.loc[12, 'fluid']) * 1e-3
    s12A = PSI("S", "T", T_0, "P", df.loc[12, 'p [bar]'] * 1e5, df.loc[12, 'fluid'])
    h_0_ambient_fluid = PSI("H", "T", T_0, "P", p_0, df.loc[11, 'fluid']) * 1e-3
    s_0_ambient_fluid = PSI("S", "T", T_0, "P", p_0, df.loc[11, 'fluid']) * 1e-3

    e11T = df.loc[11, 'h [kJ/kg]'] - h11A - T_0 * (df.loc[11, 's [J/kgK]'] - s11A) * 1e-3
    e12T = df.loc[12, 'h [kJ/kg]'] - h12A - T_0 * (df.loc[12, 's [J/kgK]'] - s12A) * 1e-3
    e11M = h11A - h_0_ambient_fluid - T_0 * (s11A - s_0_ambient_fluid)
    e12M = h12A - h_0_ambient_fluid - T_0 * (s12A - s_0_ambient_fluid)

    h_0_wf = PSI("H", "T", T_0, "P", p_0, df.loc[31, 'fluid']) * 1e-3
    s_0_wf = PSI("S", "T", T_0, "P", p_0, df.loc[31, 'fluid']) * 1e-3
    e31 = df.loc[31, 'h [kJ/kg]'] - h_0_wf - T_0 * (df.loc[31, 's [J/kgK]'] - s_0_wf) * 1e-3
    e32 = df.loc[32, 'h [kJ/kg]'] - h_0_wf - T_0 * (df.loc[32, 's [J/kgK]'] - s_0_wf) * 1e-3
    e33 = df.loc[33, 'h [kJ/kg]'] - h_0_wf - T_0 * (df.loc[33, 's [J/kgK]'] - s_0_wf) * 1e-3
    e34 = df.loc[34, 'h [kJ/kg]'] - h_0_wf - T_0 * (df.loc[34, 's [J/kgK]'] - s_0_wf) * 1e-3
    e35 = df.loc[35, 'h [kJ/kg]'] - h_0_wf - T_0 * (df.loc[35, 's [J/kgK]'] - s_0_wf) * 1e-3
    e36 = df.loc[36, 'h [kJ/kg]'] - h_0_wf - T_0 * (df.loc[36, 's [J/kgK]'] - s_0_wf) * 1e-3

    h_0_TES = PSI("H", "T", T_0, "P", p_0, df.loc[11, 'fluid']) * 1e-3
    s_0_TES = PSI("S", "T", T_0, "P", p_0, df.loc[11, 'fluid']) * 1e-3
    e21 = df.loc[21, 'h [kJ/kg]'] - h_0_TES - T_0 * (df.loc[21, 's [J/kgK]'] - s_0_TES) * 1e-3
    e22 = df.loc[22, 'h [kJ/kg]'] - h_0_TES - T_0 * (df.loc[22, 's [J/kgK]'] - s_0_TES) * 1e-3

    epsilon_EVA = (df.loc[11, 'm [kg/s]'] * (e12T - e11T) /
                   (df.loc[31, 'm [kg/s]'] * (e34 - e35) + df.loc[11, 'm [kg/s]'] * (e11M - e12M)))

    epsilon_IHX = (e36 - e35) / (e32 - e33)

    epsilon_COND = (df.loc[21, 'm [kg/s]'] * (e22 - e21)) / (df.loc[31, 'm [kg/s]'] * (e31 - e32))

    epsilon_COMP = (e31 - e36) / (df.loc[31, 'h [kJ/kg]'] - df.loc[36, 'h [kJ/kg]'])

    epsilon = {
        'EVA': epsilon_EVA,
        'COMP': epsilon_COMP,
        'COND': epsilon_COND,
        'IHX': epsilon_IHX
    }

    return epsilon


# BASE CASE
p32 = 13 * 1e5  # design variable to optimize
[df, min_td_cond] = hp_simultaneous(p32, print_results=True, plot=False, case='base', delta_t_min=5)
round(df, 5).to_csv('hp_simult_strems.csv')


# EXERGY ANALYSIS
T_0 = 283.15  # K
p_0 = 1.013e5  # Pa
epsilon = epsilon_func(T_0, p_0, df)
df_epsilon = pd.DataFrame.from_dict(epsilon, orient='index', columns=['Value'])
df_epsilon.to_csv('hp_simult_epsilon.csv')


# OPTIMIZE p32
"""
# the following method is very basic but seems to work correctly even if it takes a long time

target_min_td_cond = 5
tolerance = 0.001  # relative to min temperature difference
learning_rate = 1e4  # relative to p32

diff = abs(min_td_cond - target_min_td_cond)
step = 0

while diff > tolerance:
    # Adjust p32 based on the difference
    adjustment = (target_min_td_cond - min_td_cond) * learning_rate
    # adjustment is the smaller, the smaller the difference target_min_td_cond - min_td_cond
    p32 += adjustment

    [df, min_td_cond] = hp_simultaneous(p32, print_results=False, plot=False, case='base', delta_t_min=5)
    diff = abs(min_td_cond - target_min_td_cond)

    step += 1
    print(f"Optimization in progress, step: {step}, diff: {round(diff,5)}")

print(f"Optimization completed in {step} steps")
print("Optimized p32:", p32)


[df, min_td_cond] = hp_simultaneous(p32, print_results=True, plot=True, case='base', delta_t_min=5)
"""

# the following method doesn't work if the starting value is far away from the optimal value
'''
from scipy.optimize import minimize
target_min_td_cond = 5


def objective_function(p32):
    _, min_td_cond = hp_simultaneous(p32, print_results=False, plot=False, case='base', delta_t_min=5)
    return abs(min_td_cond - target_min_td_cond)


def numerical_gradient(p32, epsilon=1e-6):
    grad = (objective_function(p32 + epsilon) - objective_function(p32 - epsilon)) / (2 * epsilon)
    return grad


# Initial guess for p32
initial_p32 = 12.5 * 1e5

# Using BFGS algorithm for optimization
result = minimize(objective_function, initial_p32, method='BFGS', jac=numerical_gradient)

optimized_p32 = result.x
print("Optimized p32:", optimized_p32)
'''


