# Advanced Exergy Analysis of a Carnot Battery

[![forthebadge](http://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[<img src="https://raw.githubusercontent.com/oemof/tespy/9915f013c40fe418947a6e4c1fd0cd0eba45893c/docs/api/_images/logo_tespy_big.svg" alt="drawing" width="180"/>](https://github.com/oemof/tespy)

[![DOI](https://zenodo.org/badge/611892017.svg)](https://zenodo.org/doi/10.5281/zenodo.11282826)

This is a Python script that models a simple Pumped Thermal Energy Storage (high temperature HP and ORC) and performs an advanced exergetic analysis. 

This is part of my research work at [Department of Energy Engineering and Climate Protection](https://www.tu.berlin/en/energietechnik) of the Technische Universität Berlin.

## Table of contents

  - [Installation](#installation)
  - [Usage](#usage)
  - [Methodology](#Methodology)
  - [License](#license)

## Installation

[(Back to top)](#table-of-contents)

To set up the project environment, follow these steps:

1. **Install Python**: Ensure you have Python installed on your system. Version 3.9 is recommended for compatibility. You can download it from the [official Python website](https://www.python.org/downloads/).

2. **Clone the Repository**: Download the project code by cloning the repository. Open your terminal or command prompt and run:
   ```bash
   git clone https://github.com/sertomas/adex-ptes.git
   ```
   Then, navigate to the project directory:
   ```bash
   cd adex-ptes
   ```

3. **Create a Virtual Environment**: It's a good practice to create a virtual environment for Python projects. This isolates your project's dependencies from the rest of your system. Use the following command:
   ```bash
   python -m venv adex-cb
   ```
   After creating the virtual environment, you need to activate it. The activation command differs depending on your operating system:
   - On Windows:
     ```bash
     .\adex-cb\Scripts\activate
     ```
   - On Unix or MacOS:
     ```bash
     source adex-cb/bin/activate
     ```

4. **Install Required Dependencies**: The project dependencies are listed in a YAML file. However, `pip` does not directly install packages from a YAML file (commonly used with Conda environments). If you are using Conda, you can create an environment from the YAML file directly. Otherwise, for `pip`, ensure you have a `requirements.txt` file or convert the YAML content to a `pip`-compatible format. Assuming you have a `requirements.txt` file or have converted the YAML file content:
   ```bash
   pip install -r requirements.txt
   ```

Note: If you intended to include instructions for installing dependencies using a YAML file with Conda, you might need to adjust the command for installing dependencies accordingly. For example:
```bash
conda env create -f adex-ptes.yaml
```
Then, activate the Conda environment:
```bash
conda activate <env_name>
```
Ensure you replace `<env_name>` with the name of your environment as specified in the YAML file.

## Usage

[(Back to top)](#table-of-contents)

1. **Run**
   - To obtain all the results, run `hp.py` and `orc.py`. The equation system, the starting values and all the parameters are saved here. 
   - The file `functions.py` contain general functions for the modeling of the equation system
   - The results from the advanced exergy analysis of the HP and the ORC are saved in `/outputs/adex_hp` and `/outputs/adex_orc` respectively. 
   - All the diagrams created during the simulations are saved in `/outputs/diagrams`.
2. **Changes** 
   - If you want to change the fluid of the HP or of the ORC, or change the ambient conditions, you can do it in `hp.py` and `orc.py`. 
   - If you want to change the design of your subsystem, you should change the equation system in `hp.py` and `orc.py`.

## Methodology 

[(Back to top)](#table-of-contents)

1. **Model and simulate the *base case* of the HP and the ORC:**
    - The HP and the ORC of the CB are simulated using a self-made simulatenous solver.
    - The starting values, the design variables and the ambient conditions are provided. 
    - The selection of the equations to correctly simulate the system is a crucial part.
    - If necessary, the decision variables are optimized to obtain the highest efficiency.
2. **Model and simulate the *ideal case* of the HP and the ORC:**
   - **Maintain constant output:** Ensure the system's product remains is the same of the *base case*.
   - **Eliminate exergy losses:** Operate components adiabatically, except for those like condensers designed for heat dissipation.
   - **Idealize components:** Apply specific concepts to idealize all system components.
     - **Compressors and pumps:** Treated as isentropic compression processes.
     - **Turbines:** Idealized through isentropic expansion.
     - **Throttling valves:** Replaced with isentropic expanders for idealization.
     - **Heat exchangers:** Conceptualized with intermediate reversible cycles (e.g., Lorenz cycle), avoiding detailed cycle simulation.
   - **Account for additional power flows:** Adjust for power flows resulting from idealization, affecting total power consumption or fuel usage. 
3. **Model and simulate the *hybrid cases* of the HP and the ORC:**
   - **Analyze each component individually:** Consider each component in a real operation mode with others idealized, focusing on exergetic efficiency.
   - **Use enthalpy and entropy values:** Simplify analysis by avoiding direct exergy calculations.
   - **Formulate efficiency equations:** Integrate exergetic efficiency equations for standard components (compressors, expanders, heat exchangers) into the system's equation set.
4. **Correct the methods in case of errors:**
    - In case of errors (e.g. negative temperature difference), the approach should be relaxed, in order to obtain reasonable results.
## License

[(Back to top)](#table-of-contents)

MIT License (MIT). Please have a look at the [LICENSE.md](https://github.com/sertomas/adex-carnot-battery/blob/main/LICENSE.md) for more details.
