# QNetwork-RL
Agent and Environment for Queueing Network Generation using Multi-Agent RL.
The environment is used to construct a Queueing network(with priority types) sequentially with 4 actions (for a network with $n$ nodes, service distributions specified by $b$ quantiles and $p$ priority types):
1. Add Node: Tuple(Discrete(2), Box(-inf, inf, ($p$, $b$), float32))
2. Add Edge: Tuple(Discrete($2^n-2$), Box(0.0, inf, (1,), float32))
3. Edit Nodes:  Box(-inf, inf, ($n$, $p$, $b$), float32))
4. Edit Weights: Box(0.0, inf, ($n$, 1), float32)

### Requirements
SimulationPy library is required for this environment. You can install it from my [Github repository](https://github.com/NixonZ/SimulationPy).
