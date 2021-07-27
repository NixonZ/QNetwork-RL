from random import random
import numpy as np
import torch
from environment.env import Env
from agent.agent import Agent,device
from agent.Qmix import Qmix
from torch.optim import Adam
import torch.nn as nn
import copy
import pickle
from typing import List

class trainer:
    def __init__(self,p: int,b: int,environment: Env, hidden_node_dim: int, graph_dim: int, prop_steps: int,real_data,max_nodes: int = 10, buffer_size: int = 10000, lr: int = 0.0001, gamma = 0.9, epsilon = 0.1) -> None:
        
        # Agents
        self.agent_addnode = Agent("add node",(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()
        self.agent_addedge = Agent("add edge",(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()
        # self.agent_editnode = Agent("edit nodes",(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()
        # self.agent_editweights = Agent("edit weights",(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()
        self.Qmix = Qmix(2,(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()

        self.modules = nn.ModuleList( [ self.agent_addnode, self.agent_addedge, self.Qmix ] )
        self.agents = nn.ModuleList( [ self.agent_addnode, self.agent_addedge ] )
        self.w_param = nn.ModuleList( [ self.agent_addnode.action_value, self.agent_addedge.action_value,self.Qmix ] )
        self.theta_param = nn.ModuleList( [ self.agent_addnode.policy_network, self.agent_addnode.policy_network ] )

        self.env = environment
        self.max_nodes = max_nodes
        self.real_data = real_data

        self.buffer = []
        self.buffer_size = buffer_size
        self.lr = lr
        self.action_value_optimizer = Adam(filter(lambda p: p.requires_grad, self.w_param.parameters()), lr=lr)
        self.policy_optimizer = Adam(filter(lambda p: p.requires_grad, self.theta_param.parameters()), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        pass

    def learn_w(self,losses_actionvalue:List):

        self.action_value_optimizer.zero_grad()
        self.modules.zero_grad()

        loss_actionvalue = torch.zeros((1),device=device)

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
                Q_list.append(Q)

                with torch.no_grad():
                    max_Q_, _, _ = agent.action(next_state)
                    Q_list_next_state.append(max_Q_)

            Q_list = torch.cat(Q_list,dim=0)

            with torch.no_grad():
                Q_list_next_state = torch.cat(Q_list_next_state,dim=0)
                target = torch.tensor(reward,device=device) + self.gamma*(1-int(done))*self.Qmix.forward(end_state,Q_list_next_state)

            temp =  target - self.Qmix.forward(start_state,Q_list)
            loss_actionvalue += temp*temp

        loss_actionvalue.backward()
        self.action_value_optimizer.step()
        losses_actionvalue.append(loss_actionvalue.detach_().item())

    def learn_theta(self,losses_policy:List):

        self.policy_optimizer.zero_grad()
        self.modules.zero_grad()

        loss_policy = torch.zeros((1),device=device)

        for param in self.w_param.parameters():
            param.requires_grad = False

        for data in self.buffer:
            epoch = data["Epoch"]
            step = data["step"]
            done = data["done"]
            reward = data["reward"]
            start_state = data["start state"]
            end_state = data["end state"]

            Q_hat_list = []

            for agent_data in step:
                action_type = agent_data["action type"]
                agent = agent_data["agent"]
                state = agent_data["state"]
                k = agent_data["discrete action"]
                xk = agent_data["continuous action"]
                next_state = agent_data["new state"]

                Q_hat = agent.Q_hat(state)

                Q_hat_list.append(Q_hat)

            Q_hat_list = torch.cat(Q_hat_list,dim=0)

            loss_policy -=  self.Qmix.forward(start_state,Q_hat_list,True)

        loss_policy.backward()
        self.policy_optimizer.step()
        losses_policy.append(loss_policy.detach_().item())

        for param in self.w_param.parameters():
            param.requires_grad = True

    def train(self,epochs):

        done = False
        environment = copy.deepcopy(self.env)
        rewards = []

        output_folder = "./output/"
        test_name = "test1"
        savepath = output_folder + test_name

        max_reward = 0.0

        # Load Model
        try:
            checkpoint = torch.load( savepath + '/checkpoint.t7' )
            self.modules.load_state_dict(checkpoint['state_dict'])

            self.action_value_optimizer.load_state_dict(checkpoint['action_value_optimizer'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])

            losses_actionvalue = np.load( savepath + "/losses_actionvalue.npy" )
            losses_policy = np.load( savepath + "/losses_policy.npy" )
            losses_actionvalue = losses_actionvalue.tolist()
            losses_policy = losses_policy.tolist()

            i = checkpoint['epoch'] + 1
            print(i)
        except:
            i = 0
            losses_actionvalue = []
            losses_policy = []

        while i < epochs:
            temp = []
            start_state = environment.get_state_torch()
            done = False
            with torch.no_grad():
                if random() > self.epsilon:
                    # MaxQ action
                    # Add node
                    data = environment.get_state_torch()
                    _,k,xk = self.agent_addnode.action(data)
                    environment.step( ( "add node", (k,xk.cpu().detach().numpy()) ) )
                    if k == 0:
                        done = True

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

                    data = environment.get_state_torch()
                    _,k,xk = self.agent_addedge.action(data)
                    if not(done):
                        environment.step( ( "add edge", (k,xk.cpu().detach().numpy().item()) ) )
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

                else:
                    # random action

                    # Add node
                    data = environment.get_state_torch()
                    k = np.random.choice([0,1],1,p=[0.001,1-0.001])[0]
                    xk,_ = self.agent_addnode.rn_action(data,k)
                    environment.step( ( "add node", (k,xk.cpu().detach().numpy()) ) )
                    if k == 0:
                        done = True

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
                    n = environment.n
                    k = np.random.choice(list(range(0,2**(n-1)-2)),1)[0]
                    xk,_ = self.agent_addedge.rn_action(data,k)
                    if not(done):
                        environment.step( ( "add edge", (k,xk.cpu().detach().numpy().item()) ) )
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
                print(str(i)+",learning")
                self.learn_w(losses_actionvalue)
                self.learn_theta(losses_policy)
                self.buffer = []
                print("saving")
                state = {
                    'epoch': i,
                    'state_dict': self.modules.state_dict(),
                    'action_value_optimizer': self.action_value_optimizer.state_dict(),
                    'policy_optimizer': self.policy_optimizer.state_dict(),
                }
                torch.save(state, savepath + "/checkpoint.t7")
                np.save( savepath + "/losses_actionvalue.npy",losses_actionvalue)
                np.save( savepath + "/losses_policy.npy",losses_policy)

            if environment.n > self.max_nodes:
                done = True
                environment = copy.deepcopy(self.env)

            reward = environment.reward(self.real_data,0.3,100000)
            if reward > max_reward:
                max_reward = reward
                bestfile = open( savepath + "/bestenv", 'ab')
                pickle.dump({'state':environment.get_state(),'G':environment.get_state_nx()}, bestfile)
                bestfile.close()

            rewards.append(reward)
            np.save(savepath + "/rewards.npy",rewards)
            dbfile = open( savepath + "/env", 'ab')
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
            if done:
                environment = copy.deepcopy(self.env)
            i += 1