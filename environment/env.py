from typing import List, Tuple
from environment.distribution import distribution
import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
from environment.distribution import distribution


from simulationpy import Station,QNetwork,INF
class node():
    def __init__(
        self,
        service : List[distribution], # Service distribution
        C : int, # Number of servers
        num_priority : int
    ):
        self.service = service
        self.C = C
        self.num_priority = num_priority

    def convert_to_station(self):
        return Station(self.C,self.C,self.service,self.num_priority,[lambda t: INF]*self.num_priority)

class Env(gym.Env,):
    def __init__(
        self,
        arrival, # Arrival distribution
        num_priority : int, # Number of priority types
        network : List[List[Tuple]], # A list of list
        nodes_list : List[node], # list of nodes
        b : int # Number of quantile values to use
    ):
        # Environment Parameters
        self.arrival = arrival
        self.num_priority = num_priority
        self.network = network
        self.node_list = nodes_list
        self.b = b

    @property
    def action_space(self):
        n = len(self.network) # Number of nodes
        return spaces.Dict(
            {
                "add node": spaces.Tuple((
                    spaces.Discrete(2), # 0 - stop, 1 - yes
                    spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_priority,self.b)) # The Service distribution of the new node
                )),
                "add edge": spaces.Tuple((
                    spaces.Discrete(2**n-1), # Choosing the nodes to connect to the new node.
                    spaces.Box(shape=(1,), low=0, high=np.inf) # The traffic division constant.
                )),
                "edit nodes": spaces.Box(low=-1.00, high=1.00, shape=(n,self.num_priority,self.b)), # Edit the service distribution of previous present nodes
                "edit weights": spaces.Box(low=0.0, high=np.inf, shape=(n,))
            }
        )

    @property
    def obervation_space(self):
        n = len(self.network) # Number of nodes
        return spaces.Tuple((
            spaces.Box(low=-np.inf,high=np.inf,shape=(n,self.num_priority,self.b)),
            spaces.Box(low=0.0,high=1.0,shape=(n,n))
        ))

    def step(self,action: Tuple) -> Tuple[Tuple,float,bool,dict]:

        action_id = action[0]
        action_parameter = action[1]

        reward = 0.0
        done = False

        if action_id == "add node":
            if action_parameter[0] == 0:
                # Stop the generation process
                done = True
            elif action_parameter[0] == 1:
                # action_parameter[1] -> (p,b)
                service = []
                for i in range(self.num_priority):
                    quantiles = sorted(action_parameter[1][i,:])
                    for j,quantile in enumerate(quantiles):
                        quantiles[j] = (quantile, (j+1.00)/(self.b + 1.00))
                    service.append( distribution(self.b, quantiles ) )
                self.node_list.append( node(service,1,self.num_priority) )
                self.network.append([])

        if action_id == "add edge":
            chosen_edges = action_parameter[0] + 1
            i = 0
            while chosen_edges>0:
                if chosen_edges%2:
                    if len(self.network[i]):
                        for j,adj in enumerate(self.network[i]):
                            self.network[i][j] = ( adj[0] , adj[1]/( 1.00 + action_parameter[1] ) )
                        self.network[i].append( ( len(self.network) - 1, action_parameter[1]/( 1.00 + action_parameter[1] ) ) )
                    else:
                        self.network[i].append( ( len(self.network) - 1, 1.00 ) )
                chosen_edges //= 2
                i += 1
        
        if action_id == "edit nodes":
            # action_parameter -> (n,p,b)
            raise NotImplementedError

        if action_id == "edit weights":
            # action_parameter -> (n,)
            for i,_ in enumerate(action_parameter):
                total_traffic = action_parameter[i]

                for j,adj in enumerate(self.network[i]):
                    total_traffic += adj[1]*action_parameter[adj[0]]

                if total_traffic != 0:
                    for j,adj in enumerate(self.network[i]):
                        self.network[i][j] = ( adj[0] , adj[1]*( action_parameter[adj[0]] + action_parameter[i] ) / total_traffic )

        return self.get_state(), reward, done, {}

    def get_state(self):
        # Can try to return a pytorch Geometric data object also.
        n = len(self.network)
        state = []

        for node in self.node_list:
            temp = []
            for service_dist in node.service:
                temp.append([quantile[0] for quantile in service_dist.quantiles]) # Possible to include the quantile location.
            state.append(temp)
        
        adj_matrix = np.zeros((n,n),dtype=np.float32)

        for i,edge_list in enumerate(self.network):
            for adj in edge_list:
                j = adj[0]
                p_ij = adj[1]
                adj_matrix[i,j] = p_ij

        return np.array(state),adj_matrix