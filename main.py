from environment.env import Env,node
from environment.distribution import U,Exp,distribution
import numpy as np

b = 4

temp = Env(
    arrival = [lambda t: Exp(0.1)],
    num_priority= 1,
    network = [
        []
    ],
    nodes_list = [
        node( [ distribution.from_sampler(b,lambda : Exp(0.2)) ] ,1,1)
    ],
    b = b
)
print(temp.action_space)
print(temp.obervation_space)
print(temp.get_state())

quantiles = np.array([[1.09879902, 2.54394996, 4.56991541, 8.03681263]])
temp.step( ( "add node", (1,quantiles) ) )
temp.step( ( "add edge", (0, 1.00 ) ) )
temp.step( ( "add node", (1,quantiles) ) )
temp.step( ( "add edge", (2, 10 ) ) )
temp.step( ( "edit weights", [1,10,5] ) )

print()
print(temp.action_space)
print(temp.obervation_space)
print(temp.get_state())