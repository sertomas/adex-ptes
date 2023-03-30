# Declaration of global variables

# TODO: Move parameters that don't change from cb_set_pars to here

# HP and ORC configurations
delta_t_min = 5  # minimum temperature difference [K]
T_amb = 14  # temperature of ambient [°C]
p_amb = 1.013  # pressure of ambient [bar]
P_in = 1.5e6  # [e6 = MW]
P_out = 0.4e6  # [e6 = MW]

# TES configurations
V = 500  # [m^3]
rho = 1000  # [kg/m^3] constant density of water
P_min = 150 * 1e3  # [1e3 = kW], minimal power inlet or outlet
