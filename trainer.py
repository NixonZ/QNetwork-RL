import torch
from environment.env import Env
from agent.agent import Agent,device
from agent.Qmix import Qmix
from torch.optim import Adam
import torch.nn as nn
import copy
import pickle   

class trainer:
    def __init__(self,p: int,b: int,environment: Env, hidden_node_dim: int, graph_dim: int, prop_steps: int,real_data,max_nodes: int = 10, buffer_size: int = 10000,lr: int = 0.0001) -> None:
        
        # Agents
        self.agent_addnode = Agent("add node",(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()
        self.agent_addedge = Agent("add edge",(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()
        # self.agent_editnode = Agent("edit nodes",(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()
        # self.agent_editweights = Agent("edit weights",(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()
        self.Qmix = Qmix(2,(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()

        self.modules = nn.ModuleList( [ self.agent_addnode, self.agent_addedge, self.Qmix ] )
        self.agents = nn.ModuleList([ self.agent_addnode, self.agent_addedge ])

        self.env = environment
        self.max_nodes = max_nodes
        self.real_data = real_data

        self.buffer = []
        self.buffer_size = buffer_size
        self.lr = lr
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.modules.parameters()), lr=lr)
        pass

    def learn(self):

        # Train w,wmix
        loss = torch.zeros((1),device=device)

        for data in self.buffer:
            epoch = data["Epoch"]
            step = data["step"]
            done = data["done"]
            reward = data["reward"]
            start_state = data["start state"]
            end_state = data["end state"]

            Q_list = []

            Q_list_next_state = []

            for agent_data in step:
                action_type = agent_data["action type"]
                agent = agent_data["agent"]
                state = agent_data["state"]
                k = agent_data["discrete action"]
                xk = agent_data["continuous action"]
                next_state = agent_data["new state"]

                Q = agent.action_value_calc(state,k,xk)

                max_Q_, _, _ = agent.action(next_state)

                Q_list.append(Q)
                Q_list_next_state.append(max_Q_)

            Q_list = torch.tensor(Q_list,device=device)
            Q_list_next_state = torch.tensor(Q_list_next_state,device=device)
            temp =  reward + self.Qmix.forward(end_state,Q_list_next_state) - self.Qmix.forward(start_state,Q_list)
            loss += temp*temp

        loss.backward()
        self.optimizer.step()

        # Train theta
        loss = torch.zeros((1),device=device)

        for data in self.buffer:
            epoch = data["Epoch"]
            step = data["step"]
            done = data["done"]
            start_state = data["start state"]

            Q_list = []

            for agent_data in step:
                agent = agent_data["agent"]
                state = agent_data["state"]
                next_state = agent_data["new state"]

                Q_hat = agent.Q_hat(state)

                Q_list.append(Q_hat)

            Q_list = torch.tensor(Q_list,device=device,dtype=torch.float64)
            loss -=  self.Qmix.forward(start_state,Q_list)

        for param in self.Qmix.parameters():
            param.requires_grad = False
        
        for agent in self.agents:
            agent.fix_w()
        loss.backward()
        self.optimizer.step()
        for param in self.Qmix.parameters():
            param.requires_grad = True
        # self.Qmix.hypernetwork()

        for agent in self.agents:
            agent.train_w()
        pass

    def train(self,epochs):

        done = False
        environment = copy.deepcopy(self.env)

        for i in range(epochs):
            temp = []
            start_state = environment.get_state_torch()
            # Add node
            data = environment.get_state_torch()
            _,k,xk = self.agent_addnode.action(data)
            environment.step( ( "add node", (k,xk.detach().numpy()) ) )
            if k == 0:
                done = True
                environment = copy.deepcopy(self.env)
                print("Continue")
                continue

            temp.append(
                {
                    "action type" : "add node",
                    "agent" : self.agent_addnode,
                    "state" : data,
                    "discrete action" : k,
                    "continuous action" : xk,
                    "new state" : environment.get_state_torch()
                }
            )

            # Add edge
            data = environment.get_state_torch()
            _,k,xk = self.agent_addedge.action(data)
            environment.step( ( "add edge", (k,xk.detach().numpy().item()) ) )
            temp.append(
                {
                    "action type" : "add edge",
                    "agent" : self.agent_addedge,
                    "state" : data,
                    "discrete action" : k,
                    "continuous action" : xk,
                    "new state" : environment.get_state_torch()
                }
            )

            # Edit node

            # Edit weights

            if len(self.buffer) == self.buffer_size:
                print("learning")
                self.learn()
                self.buffer = []

            if environment.n > self.max_nodes:
                print("Reset")
                done = True
                environment = copy.deepcopy(self.env)

            print("simulating")
            reward = environment.reward(self.real_data)
            print(i)
            print(reward)
            dbfile = open('environment', 'ab')
            pickle.dump({'state':environment.get_state(),'G':environment.get_state_nx()}, dbfile)                     
            dbfile.close()

            self.buffer.append(
                {
                    "Epoch" : i,
                    "step" : temp,
                    "done" : done,
                    "reward" : reward,
                    "start state" : start_state,
                    "end state" : environment.get_state_torch()
                }
            )