from environment.env import Env,node
from environment.metalog import U,Exp,metalog
import numpy as np
# from agent.agent import MPNN,Graph_Representation,Agent
from trainer import trainer
from agent.Qmix import Qmix
from agent.agent import device

p = 2
b = 16
n = 6

temp = Env(
    arrival = [lambda t: Exp(0.1),lambda t: Exp(0.1)],
    num_priority= p,
    network = [
        []
    ],
    nodes_list = [
        node( [ 
            metalog.from_sampler(b,lambda : Exp(0.2),n,(0,np.inf)),
            metalog.from_sampler(b,lambda : Exp(0.23),n,(0,np.inf)) 
        ], 1, p),
    ],
    b = b,
    n_terms = n
)

# print(temp.action_space)
# print(temp.obervation_space)
# print(temp.get_state()[0].shape)

quantiles = np.array(
    [ 
        np.array(metalog.from_sampler(b,lambda : Exp(0.2),n,(0,np.inf)).quantile_val),
        np.array(metalog.from_sampler(b,lambda : Exp(0.2),n,(0,np.inf)).quantile_val)
    ])
temp.step( ( "add node", (1,quantiles) ) )
temp.step( ( "add edge", (0, 1.00 ) ) )
temp.step( ( "add node", (1,quantiles) ) )
temp.step( ( "add edge", (2, 10 ) ) )
temp.step( ( "edit weights", [1,10,5] ) )

# print()
# print(temp.action_space)
# print(temp.obervation_space)
# print(temp.get_state()[0].shape)
# print(temp.get_state_torch())

# print()

data = temp.get_state_torch()
edge_index = data.edge_index
edge_attr = data.edge_attr
x = data.x

# forward_message = MPNN((p,b),1,25,mode='forward').double()
# backward_message = MPNN((p,b),1,25,mode='backward').double()
# x = forward_message.forward(x,edge_attr,edge_index) + backward_message.forward(x,edge_attr,edge_index)

# print()
# model = Graph_Representation((p,b),1,250,500,2).double()
# print(model.forward(data))

# print()
# agent = Agent("edit weights",(p,b),1,25,50,2).double()
# print(agent.forward(data))
# print(agent)

# [ (p.numel(),p.names) for p in agent.parameters() ]

temp = trainer(p,b,temp,100,250,3,[Exp(0.07) for _ in range(10000)],buffer_size=1000)
temp.modules.to(device=device)
print(sum(p.numel() for p in temp.modules.parameters() if p.requires_grad))
temp.train(10000)

# temp = Qmix(2,(p,b),1,100,250,10).double()
# temp.set_weights(data)