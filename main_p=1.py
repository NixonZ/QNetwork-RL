from environment.env import Env,node
from environment.metalog import Exp,metalog
import numpy as np
from trainer import trainer
from agent.agent import device

b = 64
p = 1
n = 6
M = 100
temp = Env(
    arrival = [lambda t: Exp(0.1)],
    num_priority= p,
    network = [
        []
    ],
    nodes_list = [
        node( [ metalog.from_sampler(b,lambda : Exp(0.2),n,(0,np.inf)) ], 100, p)
    ],
    b = b,
    n_terms = n,
    M = M
)
quantiles = np.array([np.array(metalog.from_sampler(b,lambda : Exp(0.2),n,(0,np.inf)).quantile_val)])
temp.step( ( "add node", (M-10,quantiles) ) )
temp.step( ( "add edge", (0, 1 ) ) )

temp = trainer(p,b,M,temp,250,500,4,[Exp(0.07) for _ in range(10000)], max_nodes=4, buffer_size=10e4, train_size=10, lr= 0.0001*6, gamma = 0.9, epsilon = 0.15)
print(sum(p.numel() for p in temp.modules.parameters() if p.requires_grad))
temp.modules.to(device=device)
temp.train(10000000,"test_simple")