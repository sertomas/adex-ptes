# Advanced Exergy Analysis of a Carnot Battery

[![forthebadge](http://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[<img src="https://raw.githubusercontent.com/oemof/tespy/9915f013c40fe418947a6e4c1fd0cd0eba45893c/docs/api/_images/logo_tespy_big.svg" alt="drawing" width="180"/>](https://github.com/oemof/tespy)

This is a Python script that models a simple Carnot Battery (high temperature HP & transcritical ORC) and performs an advanced exergetic analysis. 

This is part of my research work at [Department of Energy Engineering and Environmental Protection](https://www.tu.berlin/en/energietechnik) of the Technische Universität Berlin.

## Table of contents

- [Advanced Exergy Analysis of a Carnot Battery](#advanced-exergy-analysis-of-a-carnot-battery)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)

## Installation

[(Back to top)](#table-of-contents)

1. Install Python (Version 3.9 recommended).
2. Create a virtual environment using the [environment YAML file](https://github.com/sertomas/adex-carnot-battery/blob/main/adex_carnot_battery.yaml). Clone the repository. You can use the following commands in your terminal or command prompt:
   ```bash
   git clone https://github.com/sertomas/adex-carnot-battery.git
- Navigate to the project directory
   ```bash
   cd adex-carnot-battery
- Create a virtual environment (you may use other tools like conda as well)
   ```bash
   python -m venv adex-cb
- Activate the virtual environment
   ```bash
   # On Windows
   .\adex-cb\Scripts\activate
   # On Unix or MacOS
   source adex-cb/bin/activate
- Install the required dependencies from the provided environment YAML file
   ```bash 
   pip install -r adex_carnot_battery.yaml

- Clone this repository
   ```bash
  git clone https://github.com/sertomas/adex-carnot-battery.git

## Usage

[(Back to top)](#table-of-contents)

1. **Run**
   - The `main.py` file is used to run all the simulations and perform the conventional exergy analysis and the advanced exergy analysis of the CB.
   - The thermodynamic states of the HP and the ORC from the simulation using TESPy are saved in `outputs/hp_base_results.csv` and `outputs/orc_base_results.csv`.
   - The file `func_fix.py` and `func_var.py` contain fixed/general functions as well as specific functions for the considered thermodynamic cycles.
   - The results from the conventional exergy analysis are saved in `/exan`. 
   - The results from the advanced exergy analysis of the HP and the ORC are saved in `/adex_hp` and `/adex_orc` respectively. 
   - All the diagrams created during the simulations are saved in `/diagrams`.
2. **Settings** 
   - If you want to change general parameters of the CB, you can do it in `inputs/config.json`.
   - If you want to change specific settings of the HP or ORC, you can do it in `inputs/hp_pars.json` or `inputs/orc_pars.json`.
3. **Models**
   - The models of the HP and the ORC are described in `/models/hp` and `/models/orc`. TESPy is used to perform the simulation of both thermodynamic cycles. If you want to change the structure of the models or change the specified variables, you can do it in `/models/hp/HeatPumpIhx.py` and in `/models/orc/OrcIhx.py` respectively. 
   - A change in the structure of the system can require a change in the functions for the calculation of the advanced exergy analysis in `func_var.py`. In this case, the improvement of existing functions or the implementation of new functions in `func_var.py` might be necessary too. 
4. **Advanced exergy analysis**

   The advanced exergy analysis is performed following these steps: 
   - `init_hp()` and `init_orc()` are run first to do the following tasks:
     - A DataFrame with the same stream labels of the TESPy simulations is created. 
     - The streams from and to the thermal energy storage are kept constant in each idealized system (product of the HP and fuel of the ORC) and are directly taken from the results of the TESPy simulation. 
     - The inlet ambient streams are also kept constant in each idealized system.
     - These functions are called every time the advanced exergy analysis of a (partially) idealized system is performed.
   - In `set_hp()` and `set_orc()` the system of equation is solved step-by-step.
     - This is the crucial part of the advanced exergy analysis where the operating conditions (real/ideal) of each component is considered with the use of logic variables (e.g. `COMP==True` means the compressor operates ideally). 
     - In these functions part of the equations of the entire system are solved with component-specific functions (e.g. `valve()`). 
     - At the end of these functions, the DataFrame `df_conns` with the information about all the thermodynamic states is fully defined. 
     - These functions are called every time the advanced exergy analysis of a (partially) idealized system is performed.
   - `check_balance_hp()` and `check_balance_orc()` are then used to create a DataFrame with the information about the components. 
     - Component-specific functions (e.g. `valve_bal()`) are used to check the validity of the energy balance equations.
     - The power and heat flows are calculated and saved in the DataFrame `df_comps`.
     - The generated entropy as well as the exergy destruction are calculated and saved in the DataFrame `df_comps`.
     - These functions are called every time the advanced exergy analysis of a (partially) idealized system is performed.
   - In `exan_hp()` and `exan_orc()` the exergy flows are calculated.
     - Only the physical exergy is considered.
     - These functions are called every time the advanced exergy analysis of a (partially) idealized system is performed.
   - In `perf_adex_hp()` and `perf_adex_orc()` the entire advanced exergy analyses are carried out.
     - These functions are the only ones called in `main.py`. 
     - In these functions, the above-mentioned functions are called iteratively. 
     - All possible cases are considered using the logic variables. 
     - A new DataFrame `result_df` with the results of the advanced exergy analysis is created. It contains the endogenous, exogenous and mexogenous exergy destruction terms as well as the binary interaction terms. 
     - In these functions, the minimum temperature difference in the heat exchangers can be checked with the function `qt_diagram()`. If the value is lower than the allowed minimum, the functions signalize it with a warning message. In this case, the design variable should be changed in order to avoid the error. 

## License

[(Back to top)](#table-of-contents)

MIT License (MIT). Please have a look at the [LICENSE.md](https://github.com/sertomas/adex-carnot-battery/blob/main/LICENSE.md) for more details.
