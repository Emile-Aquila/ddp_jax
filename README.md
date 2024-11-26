# ddp_jax
Differential Dynamic Programming (DDP) implementations in jax

Contents:
- Vanilla DDP
- Augmented Lagrangian DDP (ALDDP)
  - original paper: [Constrained Differential Dynamic Programming Revisited](https://ieeexplore.ieee.org/document/9561530)

<br>
 
<br>

![gif](https://github.com/Emile-Aquila/ddp_jax/blob/main/figs/output.gif)


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
