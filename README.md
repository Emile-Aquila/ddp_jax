# ddp_jax
Differential Dynamic Programming (DDP) implementations in jax

Contents:
- Vanilla DDP
- Augmented Lagrangian DDP (ALDDP)
  - [Constrained Differential Dynamic Programming Revisited](https://ieeexplore.ieee.org/document/9561530)


<br>

## Installation
check the pyproject.toml file for the required packages. You can install them using the following command:
```bash
poetry install
```

<br>

## Usage
You can run the examples using the following commands:
```bash
  poetry run python ./examples/example_ddp.py
  poetry run python ./examples/example_alddp.py
```