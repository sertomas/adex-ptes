import pandas as pd
import os

from models.hp_ihx import multiprocess_hp, serial_hp

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# fluid = 'R1233zdE'  # HIGH TEMPERATURE
# fluid = 'R1336MZZZ'  # HIGH TEMPERATURE
fluid = 'R245fa'  # LOW TEMPERATURE

config_paths = {
    'base': f'inputs/hp_ihx_{fluid}.json',
    'unavoid': f'inputs/hp_ihx_{fluid}_unavoid.json',
    'outputs': f'outputs/hp_ihx_{fluid}/'
}

# Create output directory structure if it doesn't exist
os.makedirs(config_paths['outputs'], exist_ok=True)
os.makedirs(os.path.join(config_paths['outputs'], 'streams'), exist_ok=True)
os.makedirs(os.path.join(config_paths['outputs'], 'comps'), exist_ok=True)
os.makedirs(os.path.join(config_paths['outputs'], 'diagrams'), exist_ok=True)

multi = True  # true: multiprocess, false: sequential computation

if __name__ == '__main__':
    if multi:
        multiprocess_hp(config_paths, 16e5)
    else:
        serial_hp(config_paths, 16e5)

# TEST OF ONE SINGLE SIMULATION
'''[config_test, label_test] = set_adex_hp_config('real', 'real', 'real', 'real', 'real')
[df_test, target_diff, cop_test] = solve_hp(target_p12=13e5, print_results=True, config=config_test, label=label_test, adex=False)
p12_opt = find_opt_p12(13e5, target_diff, config=config_test, label=label_test, adex=False)
[df_opt, _, _] = solve_hp(p12_opt, print_results=True, config=config_test, label=label_test, adex=False, plot=True)'''

'''[config_test, label_test] = set_adex_hp_config('real', 'real', 'real', 'real', 'real')
[df_test, target_diff, cop_test] = solve_hp(target_p12=13e5, print_results=True, config=config_test, label=label_test, adex=False)
p12_opt = find_opt_p12(13e5, target_diff, config=config_test, label=label_test, adex=False)
[df_opt, _, _] = solve_hp(p12_opt, print_results=True, config=config_test, label=label_test, adex=False, plot=True)
'''