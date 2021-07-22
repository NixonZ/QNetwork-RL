from typing import List, Tuple
import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
from environment.metalog import metalog

from simulationpy import Station,QNetwork,INF

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

import networkx as nx

import pandas as pd

import time

from agent.agent import device

def call_event_type_list(event_type_list,t):
    events = []
    for event in event_type_list:
        events.append(event(t))
    return events

# class node():
#     def __init__(
#         self,
#         service : List[distribution], # Service distribution
#         C : int, # Number of servers
#         num_priority : int
#     ):
#         assert( len(service) == num_priority )
#         self.service = service
#         self.C = C
#         self.num_priority = num_priority

#     def convert_to_station(self):
#         service_dist = [ lambda t: dist.sampler(kind='piecewise linear') for dist in self.service ]
#         return Station(self.C,self.C,service_dist,self.num_priority,[lambda t: INF]*self.num_priority)
    
#     def edit_service(self,new_quantiles):
#         # new_quantiles -> (p,b)
#         for i in range(self.num_priority):
#             b = self.service[i].b
#             quantiles = new_quantiles[i,:] # (b,1)
#             quantiles = quantiles.reshape((b,1))
#             self.service[i] = distribution(b,quantiles) # Either directly substitute or add subtract.

# class Env(gym.Env,):
#     def __init__(
#         self,
#         arrival : List, # Arrival distribution
#         num_priority : int, # Number of priority types
#         network : List[List[Tuple]], # A list of list
#         nodes_list : List[node], # list of nodes
#         b : int # Number of quantile values to use
#     ):
#         # Environment Parameters
#         self.arrival = arrival
#         self.num_priority = num_priority
#         self.network = network
#         self.node_list = nodes_list
#         self.b = b

#     @property
#     def action_space(self):
#         n = len(self.network) # Number of nodes
#         return spaces.Dict(
#             {
#                 "add node": spaces.Tuple((
#                     spaces.Discrete(2), # 0 - stop, 1 - yes
#                     spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_priority,self.b)) # The Service distribution of the new node
#                 )),
#                 "add edge": spaces.Tuple((
#                     spaces.Discrete(2**n-1), # Choosing the nodes to connect to the new node.
#                     spaces.Box(shape=(1,), low=0, high=np.inf) # The traffic division constant.
#                 )),
#                 "edit nodes": spaces.Box(low=-1.00, high=1.00, shape=(n,self.num_priority,self.b)), # Edit the service distribution of previous present nodes
#                 "edit weights": spaces.Box(low=0.0, high=np.inf, shape=(n,))
#             }
#         )

#     @property
#     def obervation_space(self):
#         n = len(self.network) # Number of nodes
#         return spaces.Tuple((
#             spaces.Box(low=-np.inf,high=np.inf,shape=(n,self.num_priority,self.b)),
#             spaces.Box(low=0.0,high=1.0,shape=(n,n))
#         ))

#     @property
#     def n(self):
#         return len(self.network)
    
#     def step(self,action: Tuple) -> Tuple[Tuple,float,bool,dict]:

#         action_id = action[0]
#         action_parameter = action[1]

#         reward = 0.0
#         done = False

#         if action_id == "add node":
#             if action_parameter[0] == 0:
#                 # Stop the generation process
#                 done = True
#             elif action_parameter[0] == 1:
#                 # action_parameter[1] -> (p,b)
#                 service = []
#                 for i in range(self.num_priority):
#                     quantiles = sorted(action_parameter[1][i,:])
#                     for j,quantile in enumerate(quantiles):
#                         quantiles[j] = (quantile, (j+1.00)/(self.b + 1.00))
#                     service.append( distribution(self.b, quantiles ) )
#                 self.node_list.append( node(service,1,self.num_priority) )
#                 self.network.append([])

#         if action_id == "add edge":
#             chosen_edges = action_parameter[0] + 1
#             i = 0
#             while chosen_edges>0:
#                 if chosen_edges%2:
#                     if len(self.network[i]):
#                         for j,adj in enumerate(self.network[i]):
#                             self.network[i][j] = ( adj[0] , adj[1]/( 1.00 + action_parameter[1] ) )
#                         self.network[i].append( ( len(self.network) - 1, action_parameter[1]/( 1.00 + action_parameter[1] ) ) )
#                     else:
#                         self.network[i].append( ( len(self.network) - 1, 1.00 ) )
#                 chosen_edges //= 2
#                 i += 1
        
#         if action_id == "edit nodes":
#             # action_parameter -> (n,p,b)
#             n = len(self.network)
#             for i in range(n):
#                 new_quantiles = action_parameter[i]
#                 self.node_list[i].edit_service(new_quantiles)

#         if action_id == "edit weights":
#             # action_parameter -> (n,)
#             for i,_ in enumerate(action_parameter):
#                 total_traffic = action_parameter[i]

#                 for j,adj in enumerate(self.network[i]):
#                     total_traffic += adj[1]*action_parameter[adj[0]]

#                 if total_traffic != 0:
#                     for j,adj in enumerate(self.network[i]):
#                         self.network[i][j] = ( adj[0] , adj[1]*( action_parameter[adj[0]] + action_parameter[i] ) / total_traffic )

#         return self.get_state(), reward, done, {}

#     def get_state(self):
#         # Can try to return a pytorch Geometric data object also.
#         n = len(self.network)
#         state = []

#         for node in self.node_list:
#             temp = []
#             for service_dist in node.service:
#                 temp.append(np.array(service_dist.quantile_val)) # Possible to include the quantile location.
#             state.append(np.array(temp))
        
#         adj_matrix = np.zeros((n,n),dtype=np.float32)

#         for i,edge_list in enumerate(self.network):
#             for adj in edge_list:
#                 j = adj[0]
#                 p_ij = adj[1]
#                 adj_matrix[i,j] = p_ij

#         return np.array(state),adj_matrix
    
#     def get_state_nx(self) -> nx.DiGraph:
#         G = nx.DiGraph()
#         for i,node in enumerate(self.node_list):
#             node_data = []
#             for service_dist in node.service:
#                 node_data.append(service_dist.quantile_val)
#             G.add_node(i, x = node_data)
#         for i,adj in enumerate(self.network):
#             for pair in adj:
#                 G.add_edge(i,pair[0],edge_attr=pair[1])
#         return G

#     def get_state_torch(self) -> Data:
#         return from_networkx(self.get_state_nx())

#     def simulate(self,max_events = 10000):
#         station_list = [ node.convert_to_station() for node in self.node_list ]
#         patience_time = [ lambda t: INF ]*self.num_priority
#         arrival_processes = self.arrival
#         start = time.time()
#         Timer = True
#         System = QNetwork(0,0,self.network,station_list,patience_time)

#         discrete_events = 0
#         arriving_customer = 0
#         t = 0.0

#         ta = call_event_type_list(arrival_processes,t)

#         waiting_time = []
#         System.initialize_CSV("./simulation_data")


#         System.logger(t)

#         while discrete_events < max_events:
#             least_station_index,least_dep_time = System.find_least_dep_time()

#             t = np.min( [least_dep_time] + ta )
#             System.server_updates(t)

#             if t == np.min(ta):

#                 # arrival happening
#                 priority = np.argmin(ta)
#                 System.add_customer_to_graph(t, [priority,arriving_customer])
#                 # System.add_customer_to_graph_vir(t, [priority,arriving_customer],True,arrival_processes,ta)
#                 arriving_customer += 1
#                 ta[priority] = t + arrival_processes[priority](t)
#             else:
#                 System.departure_updates(least_station_index,t)
#             if discrete_events%(max_events//10) == 0:
#                 System.dump_counter_variable_memory("./simulation_data")
#             discrete_events += 1
#             if time.time() - start > 10.0:
#                 Timer = False
#         return Timer
    
#     def reward(self,real_data,sigma=0.3,max_events = 10000):
#         Timer = self.simulate(max_events)
#         if not(Timer):
#             return -10.0
#         data = pd.read_csv("./simulation_data.csv")
#         del data[data.columns[-1]]
#         data = data[ data['Wait time'] >=0 ]['Wait time'].values
#         dist1 = distribution.from_data(self.b,data)
#         dist2 = distribution.from_data(self.b,real_data)

#         tot_var = dist1.distance(dist2)
#         # return np.exp( -1*( (tot_var*tot_var)/(2*sigma*sigma) ) )/(sigma*np.sqrt(2*np.pi))
#         return 1/tot_var

#     def draw(self):
#         nx.draw(self.get_state_nx(),with_labels=True)

class node():
    def __init__(
        self,
        service : List[metalog], # Service distribution
        C : int, # Number of servers
        num_priority : int
    ):
        assert( len(service) == num_priority )
        self.service = service
        self.C = C
        self.num_priority = num_priority

    def convert_to_station(self):
        service_dist = [ lambda t: dist.sampler(kind='metalog') for dist in self.service ]
        return Station(self.C,self.C,service_dist,self.num_priority,[lambda t: INF]*self.num_priority)
    
    def edit_service(self,new_quantiles):
        # new_quantiles -> (p,b)
        for i in range(self.num_priority):
            b = self.service[i].b
            n_terms = self.service[i].n_terms
            quantiles = new_quantiles[i,:] # (b,1)
            quantiles = quantiles.reshape((b,1))
            self.service[i] = metalog(b,quantiles,n_terms,(0,np.inf)) # Either directly substitute or add subtract.

class Env(gym.Env,):
    def __init__(
        self,
        arrival : List, # Arrival distribution
        num_priority : int, # Number of priority types
        network : List[List[Tuple]], # A list of list
        nodes_list : List[node], # list of nodes
        b : int, # Number of quantile values to use
        n_terms : int # Number of terms to use
    ):
        # Environment Parameters
        self.arrival = arrival
        self.num_priority = num_priority
        self.network = network
        self.node_list = nodes_list
        self.b = b
        self.n_terms = n_terms

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

    @property
    def n(self):
        return len(self.network)
    
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
                    service.append( metalog(self.b, quantiles,self.n_terms,(0,np.inf) ) )
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
            n = len(self.network)
            for i in range(n):
                new_quantiles = action_parameter[i]
                self.node_list[i].edit_service(new_quantiles)

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
                temp.append(np.array(service_dist.quantile_val)) # Possible to include the quantile location.
            state.append(np.array(temp))
        
        adj_matrix = np.zeros((n,n),dtype=np.float32)

        for i,edge_list in enumerate(self.network):
            for adj in edge_list:
                j = adj[0]
                p_ij = adj[1]
                adj_matrix[i,j] = p_ij

        return np.array(state),adj_matrix
    
    def get_state_nx(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for i,node in enumerate(self.node_list):
            node_data = []
            for service_dist in node.service:
                node_data.append(service_dist.quantile_val)
            G.add_node(i, x = node_data)
        for i,adj in enumerate(self.network):
            for pair in adj:
                G.add_edge(i,pair[0],edge_attr=pair[1])
        return G

    def get_state_torch(self) -> Data:
        return from_networkx(self.get_state_nx()).to(device)

    def simulate(self,max_events = 10000):
        station_list = [ node.convert_to_station() for node in self.node_list ]
        patience_time = [ lambda t: INF ]*self.num_priority
        arrival_processes = self.arrival
        start = time.time()
        Timer = True
        System = QNetwork(0,0,self.network,station_list,patience_time)

        discrete_events = 0
        arriving_customer = 0
        t = 0.0

        ta = call_event_type_list(arrival_processes,t)

        waiting_time = []
        System.initialize_CSV("./simulation_data")


        System.logger(t)

        while discrete_events < max_events:
            least_station_index,least_dep_time = System.find_least_dep_time()

            t = np.min( [least_dep_time] + ta )
            System.server_updates(t)

            if t == np.min(ta):

                # arrival happening
                priority = np.argmin(ta)
                System.add_customer_to_graph(t, [priority,arriving_customer])
                # System.add_customer_to_graph_vir(t, [priority,arriving_customer],True,arrival_processes,ta)
                arriving_customer += 1
                ta[priority] = t + arrival_processes[priority](t)
            else:
                System.departure_updates(least_station_index,t)
            if discrete_events%(max_events//10) == 0:
                System.dump_counter_variable_memory("./simulation_data")
            discrete_events += 1
            if time.time() - start > 10.0:
                Timer = False
        return Timer
    
    def reward(self,real_data,sigma=0.3,max_events = 10000):
        Timer = self.simulate(max_events)
        if not(Timer):
            return -10.0
        data = pd.read_csv("./simulation_data.csv")
        del data[data.columns[-1]]
        data = data[ data['Wait time'] >=0 ]['Wait time'].values
        dist1 = metalog.from_data(self.b,data,self.n_terms,(0,np.inf) )
        dist2 = metalog.from_data(self.b,real_data,self.n_terms,(0,np.inf) )

        tot_var = dist1.distance(dist2)
        # return np.exp( -1*( (tot_var*tot_var)/(2*sigma*sigma) ) )/(sigma*np.sqrt(2*np.pi))
        return 1/tot_var

    def draw(self):
        nx.draw(self.get_state_nx(),with_labels=True)