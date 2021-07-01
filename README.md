# QNetwork-RL
Agent and Environment for Queueing Network Generation using Multi-Agent RL.
The environment is used to construct a Queueing network(with priority types) sequentially with 4 actions (for a network with ![equation](https://latex.codecogs.com/svg.latex?n) nodes, service distributions specified by ![equation](https://latex.codecogs.com/svg.latex?b) quantiles and ![equation](https://latex.codecogs.com/svg.latex?p) priority types):
1. Add Node: Tuple(Discrete(2), Box(-inf, inf, (![equation](https://latex.codecogs.com/svg.latex?p), ![equation](https://latex.codecogs.com/svg.latex?b)), float32))
2. Add Edge: Tuple(Discrete(![equation](https://latex.codecogs.com/svg.latex?2%5En-2)), Box(0.0, inf, (1,), float32))
3. Edit Nodes:  Box(-inf, inf, (![equation](https://latex.codecogs.com/svg.latex?n), ![equation](https://latex.codecogs.com/svg.latex?p), ![equation](https://latex.codecogs.com/svg.latex?b)), float32))
4. Edit Weights: Box(0.0, inf, (![equation](https://latex.codecogs.com/svg.latex?n), 1), float32)

### Requirements
SimulationPy library is required for this environment. You can install it from my [Github repository](https://github.com/NixonZ/SimulationPy).
