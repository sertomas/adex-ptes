# Advanced Exergy Analysis of a Carnot Battery

[![forthebadge](http://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[<img src="https://raw.githubusercontent.com/oemof/tespy/9915f013c40fe418947a6e4c1fd0cd0eba45893c/docs/api/_images/logo_tespy_big.svg" alt="drawing" width="180"/>](https://github.com/oemof/tespy)

A Python script that models a simple Carnot Battery (HP & ORC) and performs an advanced exergetic analysis. 

This is part of my Master Thesis at Technische Universität Berlin ([Department of Energy Engineering and Environmental Protection](https://www.tu.berlin/en/energietechnik)).

## Table of contents

- [Advanced Exergy Analysis of a Carnot Battery](#advanced-exergy-analysis-of-a-carnot-battery)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)

## Installation

[(Back to top)](#table-of-contents)

1. Install Python (I'm currently using 3.8.16).

2. Create an environment using the [environment YAML file](https://github.com/sertomas/adex-carnot-battery/blob/main/adex_carnot_battery.yaml).

3. Clone this repository

## Usage

[(Back to top)](#table-of-contents)

1. Run `cb_main` to create the model of the Carnot battery.
   
    *If you want to change the parameters of the HP or the ORC, you can do it in `cb_set_pars`.*

    *If you want to change the structure of the systems, you can do it in `cb_network`.*

1. Run `adv_exan` to perform an advanced exergetic analysis of the Carnot battery. 

## License

[(Back to top)](#table-of-contents)

MIT License (MIT). Please have a look at the [LICENSE.md](https://github.com/sertomas/adex-carnot-battery/blob/main/LICENSE.md) for more details.
