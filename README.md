[![CI](https://github.com/SH2282000/State-Vector-Simulator/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SH2282000/State-Vector-Simulator/actions/workflows/ci.yml)

I can help you create a comprehensive `README.md` for your project. Here is a draft based on the information you provided:

# State Vector Simulator

Project of a state vector simulator developed in the context of the Quantum Computing practicum of LMU.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

## Introduction

The State Vector Simulator is a project developed as part of the Quantum Computing practicum at LMU. It aims to simulate quantum states and operations using the OpenQASM and Python programming languages.

## Features

- Quantum state simulation
- Quantum gate operations
- Measurement of quantum states

## Installation

To install the State Vector Simulator, follow these steps:

1. Clone the repository:
 ```sh
 git clone https://github.com/SH2282000/State-Vector-Simulator.git
 ```
2. Navigate to the project directory:
 ```sh
 cd State-Vector-Simulator
 ```
3. Install `[poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)`.
4. Install the necessary dependencies:
 ```sh
 poetry install
 ```
5. Enter the shell
```sh
poetry shell
```
6. Run the custom setup for use w/ `cython`
```sh
poetry run python setup.py build_ext --inplace
```

## Usage

To use the State Vector Simulator, follow these steps:

1. Run the simulator:
```python
c = parseQCP(file)
simulator = SVtemplate(c)
simulator.simulate()
```
Please take a look at the `tests/test_template.py` for more insights on how to use the simulator.


## Contributors

- Shannah Santucci
- Victor
- Santo Thies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

You can copy this content into your `README.md` file. If you need any adjustments or additional information, please let me know!
