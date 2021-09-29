from random import random
import numpy as np
import torch
from environment.env import Env
from environment.metalog import metalog
from agent.agent import Agent,device
from agent.Qmix import Qmix
from torch.optim import Adam
import torch.nn as nn
import copy
import pickle
from typing import List

class trainer:
    def __init__(self,
        p: int,
        b: int,
        M: int,
        environment: Env, 
        hidden_node_dim: int, 
        graph_dim: int, 
        prop_steps: int,
        real_data,
        arrival_data,
        max_nodes: int = 10, 
        buffer_size: int = 50000,
        train_size: int = 100, 
        lr: int = 0.0001, 
        gamma = 0.9, 
        epsilon = 0.1) -> None:
        
        # Agents
        self.agent_addnode = Agent("add node",(p,b),M,1,hidden_node_dim,graph_dim,prop_steps).double()
        self.agent_addedge = Agent("add edge",(p,b),M,1,hidden_node_dim,graph_dim,prop_steps).double()
        self.agent_editnode = Agent("edit nodes",(p,b),M,1,hidden_node_dim,graph_dim,prop_steps).double()
        self.agent_editweights = Agent("edit weights",(p,b),M,1,hidden_node_dim,graph_dim,prop_steps).double()
        self.M = M
        self.Qmix = Qmix(4,(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()

        self.modules = nn.ModuleList( [ self.agent_addnode, self.agent_addedge, self.agent_editnode, self.agent_editweights, self.Qmix ] )
        self.agents = nn.ModuleList( [ self.agent_addnode, self.agent_addedge, self.agent_editnode, self.agent_editweights ] )

        self.w_param = nn.ModuleList( [ self.agent_addnode.w_param, self.agent_addedge.w_param, self.agent_editnode.w_param, self.agent_editweights.w_param, self.Qmix ] )
        self.theta_param = nn.ModuleList( [ self.agent_addnode.theta_param, self.agent_addnode.theta_param, self.agent_editnode.theta_param, self.agent_editweights.theta_param ] )

        self.env = environment
        self.max_nodes = max_nodes
        self.real_dist = metalog.from_data(b,real_data,6,(0,np.inf) )
        self.arrival_data = arrival_data
        # self.real_data = real_data

        self.buffer = []
        self.buffer_size = buffer_size
        self.train_size = train_size
        self.lr = lr
        self.action_value_optimizer = Adam(filter(lambda p: p.requires_grad, self.w_param.parameters()), lr=lr)
        self.policy_optimizer = Adam(filter(lambda p: p.requires_grad, self.theta_param.parameters()), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        pass

    def sample(self,size: int = 100):
        p = []
        sum = 0.0
        for i in range(len(self.buffer)):
            p.append(0.5**i)
            sum += p[-1]
        p = [prio/sum for prio in p]
        buffer = sorted(self.buffer, key = lambda experience: abs(experience["reward"]))
        buffer = np.random.choice(buffer, size=size, p=p)
        return buffer

    def learn_w(self,sample,losses_actionvalue:List):

        self.action_value_optimizer.zero_grad()
        self.modules.zero_grad()

        loss_actionvalue = torch.zeros((1),device=device)

        for data in sample:
            epoch = data["Epoch"]
            step = data["step"]
            done = data["done"]
            reward = data["reward"]
            start_state = copy.deepcopy(data["start state"]).to(device)
            end_state = copy.deepcopy(data["end state"]).to(device)

            Q_list = []

            Q_list_next_state = []


            for agent_data in step:
                action_type = agent_data["action type"]
                agent = agent_data["agent"]
                state = copy.deepcopy(agent_data["state"]).to(device)
                k = agent_data["discrete action"]
                xk = copy.deepcopy(agent_data["continuous action"]).to(device)
                next_state = copy.deepcopy(agent_data["new state"]).to(device)

                Q = agent.action_value_calc(state,k,xk)
                Q_list.append(Q)

                with torch.no_grad():
                    max_Q_, _, _ = agent.action(next_state,done)
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

    def learn_theta(self,sample,losses_policy:List):

        self.policy_optimizer.zero_grad()
        self.modules.zero_grad()

        loss_policy = torch.zeros((1),device=device)

        for param in self.w_param.parameters():
            param.requires_grad = False

        for data in sample:
            epoch = data["Epoch"]
            step = data["step"]
            done = data["done"]
            reward = data["reward"]
            start_state = copy.deepcopy(data["start state"]).to(device)
            end_state = data["end state"]

            Q_hat_list = []

            for agent_data in step:
                action_type = agent_data["action type"]
                agent = agent_data["agent"]
                state = copy.deepcopy(agent_data["state"]).to(device)
                k = agent_data["discrete action"]
                xk = agent_data["continuous action"]
                next_state = agent_data["new state"]

                Q_hat = agent.Q_hat(state,done)

                Q_hat_list.append(Q_hat)

            Q_hat_list = torch.cat(Q_hat_list,dim=0)

            loss_policy -=  self.Qmix.forward(start_state,Q_hat_list,True)

        loss_policy.backward()
        self.policy_optimizer.step()
        losses_policy.append(loss_policy.detach_().item())

        for param in self.w_param.parameters():
            param.requires_grad = True

    def train(self,epochs,test_name):

        done = False
        environment = copy.deepcopy(self.env)

        output_folder = "./output/"
        test_name = test_name
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

            rewards = np.load( savepath + "/rewards.npy")
            end_state_rewards = np.load( savepath + "/end_state_rewards.npy")
            rewards = rewards.tolist()
            end_state_rewards = end_state_rewards.tolist()

            i = checkpoint['epoch'] + 1
            print(i)
        except:
            i = 0
            losses_actionvalue = []
            losses_policy = []
            rewards = []
            end_state_rewards = []
            print(i)

        while i < epochs:
            temp = []
            start_state = environment.get_state_torch()
            done = False
            with torch.no_grad():
                if random() > self.epsilon:
                    # MaxQ action

                    # Add node
                    data = environment.get_state_torch()
                    _,k,xk = self.agent_addnode.action(data.to(device),done)
                    environment.step( ( "add node", (k,xk.cpu().detach().numpy()) ) )
                    if k == 0:
                        done = True

                    temp.append(
                        {
                            "action type" : "add node",
                            "agent" : self.agent_addnode,
                            "state" : data.cpu(),
                            "discrete action" : k,
                            "continuous action" : xk.cpu(),
                            "new state" : environment.get_state_torch().cpu()
                        }
                    )

                    # Add Edge
                    data = environment.get_state_torch()
                    _,k,xk = self.agent_addedge.action(data.to(device),done)
                    if not(done):
                        environment.step( ( "add edge", (k,xk.cpu().detach().numpy().item()) ) )
                    temp.append(
                        {
                            "action type" : "add edge",
                            "agent" : self.agent_addedge,
                            "state" : data.cpu(),
                            "discrete action" : k,
                            "continuous action" : xk.cpu(),
                            "new state" : environment.get_state_torch()
                        }
                    )

                    # Edit node
                    data = environment.get_state_torch()
                    _,k,xk = self.agent_editnode.action(data.to(device),done)
                    environment.step( ( "edit node", (k,xk.cpu().detach().numpy()) ) )
                    temp.append(
                        {
                            "action type" : "edit node",
                            "agent" : self.agent_editnode,
                            "state" : data.cpu(),
                            "discrete action" : k,
                            "continuous action" : xk.cpu(),
                            "new state" : environment.get_state_torch()
                        }
                    )

                    # Edit weights
                    data = environment.get_state_torch()
                    _,k,xk = self.agent_editweights.action(data.to(device),done)
                    environment.step( ( "edit weights", (k,xk.cpu().detach().numpy().item()) ) )
                    temp.append(
                        {
                            "action type" : "edit weights",
                            "agent" : self.agent_editweights,
                            "state" : data.cpu(),
                            "discrete action" : k,
                            "continuous action" : xk.cpu(),
                            "new state" : environment.get_state_torch()
                        }
                    )

                else:
                    # random action

                    # Add node
                    data = environment.get_state_torch()
                    k = np.random.choice(list(range(0,self.M+1)),1,p=[0.001]+[(1-0.001)/self.M]*self.M)[0]
                    xk,_ = self.agent_addnode.rn_action(data.to(device),k)
                    environment.step( ( "add node", (k,xk.cpu().detach().numpy()) ) )
                    if k == 0:
                        done = True

                    temp.append(
                        {
                            "action type" : "add node",
                            "agent" : self.agent_addnode,
                            "state" : data.cpu(),
                            "discrete action" : k,
                            "continuous action" : xk.cpu(),
                            "new state" : environment.get_state_torch().cpu()
                        }
                    )
                    
                    # Add edge
                    data = environment.get_state_torch()
                    n = environment.n
                    if not(done):
                        k = np.random.choice(list(range(0,2**(n-1)-2)),1)[0]
                    else:
                        k = np.random.choice(list(range(0,2**n-2)),1)[0]
                    xk,_ = self.agent_addedge.rn_action(data.to(device),k)
                    if not(done):
                        environment.step( ( "add edge", (k,xk.cpu().detach().numpy().item()) ) )
                    temp.append(
                        {
                            "action type" : "add edge",
                            "agent" : self.agent_addedge,
                            "state" : data.cpu(),
                            "discrete action" : k,
                            "continuous action" : xk.cpu(),
                            "new state" : environment.get_state_torch().cpu()
                        }
                    )

                    # Edit node
                    data = environment.get_state_torch()
                    n = environment.n
                    k = np.random.choice(list(range(0,n)),1)[0]
                    xk,_ = self.agent_editnode.rn_action(data.to(device),k)
                    environment.step( ( "edit node", (k,xk.cpu().detach().numpy()) ) )
                    temp.append(
                        {
                            "action type" : "edit node",
                            "agent" : self.agent_editnode,
                            "state" : data.cpu(),
                            "discrete action" : k,
                            "continuous action" : xk.cpu(),
                            "new state" : environment.get_state_torch()
                        }
                    )

                    # Edit weights
                    data = environment.get_state_torch()
                    n = environment.n
                    i_ = np.random.choice(list(range(0,n-1)),1)[0]
                    j_ = np.random.choice(list(range(i_+1,n)),1)[0]
                    k = [i_,j_]

                    xk,_ = self.agent_editweights.rn_action(data.to(device),k)
                    environment.step( ( "edit weights", (k,xk.cpu().detach().numpy().item()) ) )
                    temp.append(
                        {
                            "action type" : "edit weights",
                            "agent" : self.agent_editweights,
                            "state" : data.cpu(),
                            "discrete action" : k,
                            "continuous action" : xk.cpu(),
                            "new state" : environment.get_state_torch()
                        }
                    )

            if len(self.buffer) % self.train_size == 0 and len(self.buffer) >0:
                print(str(i)+",learning")
                sample = self.sample(self.train_size)
                self.learn_w(sample,losses_actionvalue)
                self.learn_theta(sample,losses_policy)
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

            reward = environment.reward(self.arrival_data,self.real_dist,0.3,10000,test_name)
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
                    "start state" : start_state.cpu(),
                    "end state" : environment.get_state_torch().cpu()
                }
            )
            if len(self.buffer) > self.buffer_size:
                # Limit buffer to only last buffer size values
                self.buffer = self.buffer[-self.buffer_size:]
        
            if done:
                end_state_rewards.append(reward)
                np.save(savepath + "/end_state_rewards.npy",end_state_rewards)
                environment = copy.deepcopy(self.env)
            i += 1