from environment.env import Env,node
from environment.metalog import Exp,metalog
import numpy as np
from trainer import trainer
from agent.agent import device
import copy
import pandas as pd

b = 32
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
        node( [ metalog.from_sampler(b,lambda : Exp(0.2),n,(0,np.inf)) ], M, p)
    ],
    b = b,
    n_terms = n,
    M = M
)

quantiles = np.array([np.array(metalog.from_sampler(b,lambda : Exp(0.2),n,(0,np.inf)).quantile_val)])
temp.step( ( "add node", (M-10,quantiles) ) )
temp.step( ( "add edge", (0, 1 ) ) )
seed_env = copy.deepcopy(temp)
temp.step( ( "add node", (M-50,quantiles) ) )
temp.step( ( "add edge", (2, 10 ) ) )
temp.step( ( "edit weights", [[0,2],0.9] ) )

temp.simulate(10000,"real_data")
data = pd.read_csv("./output/real_data.csv")

arrival_data = data[ data['Wait time'] >=0 ][['Priority Level','Time of arrival']]
real_data = data[ data['Wait time'] >=0 ]['Wait time'].values

temp = trainer(p,b,M,seed_env,250,500,4,real_data,arrival_data, max_nodes=5, buffer_size=10e4, train_size=100, lr= 0.0001*6, gamma = 0.9, epsilon = 0.15)
print(sum(p.numel() for p in temp.modules.parameters() if p.requires_grad))
temp.modules.to(device=device)
temp.train(10000000,"test1")