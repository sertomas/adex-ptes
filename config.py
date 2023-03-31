# Declaration of global variables

# TODO: Move parameters that don't change from cb_set_pars to here

# working fluids:
fluids = ['R245fa', 'water']

# ambient conditions:
T_amb = 10  # temperature of ambient [°C]
p_amb = 1.013  # pressure of ambient [bar]

# dimension parameters of ORC and HP
P_in = 1.5e6  # [e6 = MW]
P_out = 0.4e6  # [e6 = MW]
delta_t_min = 5  # minimum temperature difference [K]

# dimension parameters for TES
V = 1000  # [m^3] 500 daily storage, 5000 weekly storage, 50000 monthly storage
rho = 1000  # [kg/m^3] constant density of water
P_min = 150 * 1e3  # [1e3 = kW], minimal power inlet or outlet
