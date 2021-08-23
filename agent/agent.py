import torch
from torch.jit.frontend import NotSupportedError
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

import numpy as np

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()

class MPNN(MessagePassing):

    def __init__(self,node_embedding_dim,edge_embedding_dim,hidden_node_dim,mode = 'forward'):
        super(MPNN,self).__init__(aggr="add",flow = 'source_to_target' if mode == 'forward' else 'target_to_source',node_dim=0)

        p = node_embedding_dim[0]
        b = node_embedding_dim[1]

        self.reduce = nn.Sequential(
            nn.Conv2d(2,10,(p,1),padding=(p,0)),
            nn.Conv2d(10,50,(p,b//2+1)),
            nn.Conv2d(50,5,(p,b//4),stride=(1,b//16)),
            nn.Flatten(start_dim=1),
            nn.Linear(5*3*5, hidden_node_dim)
        )

        self.node_data = nn.ModuleList( [ nn.Linear(hidden_node_dim + edge_embedding_dim,b) for _ in range(p) ] )

    def forward(self,x,edge_attr,edge_index):
        '''
        x : [|V|, node_embedding_dim]
        edge_attr : [|E|, edge_embedding_dim]
        edge_index : [2,|E|]
        '''
        return self.propagate(edge_index, x = x, edge_attr = edge_attr)

    def message(self,x_i,x_j,edge_attr):
        temp = torch.cat((x_i.unsqueeze(1),x_j.unsqueeze(1)),dim=1)
        x = self.reduce.forward(temp)
        x = torch.cat((x,edge_attr.unsqueeze(-1)),dim=1)
        temp = []
        for i in range(self.node_data.__len__()):
            temp.append(self.node_data[i].forward(x).unsqueeze(1))
        return torch.cat(temp,dim=1).to(device)
    
class Graph_Representation(nn.Module):

    def __init__(self,node_embedding_dim,edge_embedding_dim,hidden_node_dim,graph_dim = 50,prop_steps = 2):
        super(Graph_Representation,self).__init__()

        p = node_embedding_dim[0]
        b = node_embedding_dim[1]

        # Message Passing Layers
        self.prop_steps = prop_steps
        self.forward_message = MPNN(node_embedding_dim,edge_embedding_dim,hidden_node_dim,mode='forward')
        self.backward_message = MPNN(node_embedding_dim,edge_embedding_dim,hidden_node_dim,mode='backward')
        # self.MPN_list = nn.ModuleList( [ MPNN(node_embedding_dim,edge_embedding_dim,hidden_node_dim) for _ in range(prop_steps) ] )

        # Node Dimensionality reduction
        self.reduce = nn.Sequential(
            nn.Conv2d(1,10,(p,1),padding=(p,0)),
            nn.Conv2d(10,50,(p,b//2+1)),
            nn.Conv2d(50,5,(p,b//4),stride=(1,b//16)),
            nn.Flatten(start_dim=1),
            nn.Linear(5*3*5, hidden_node_dim)
        )

        # Learning Graph representation Layers
        self.gm = nn.Linear(hidden_node_dim,graph_dim)
        self.fm = nn.Linear(hidden_node_dim,graph_dim)

    def forward(self,batch:Data):
        '''
        batch : Batch
        '''
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        for i in range(self.prop_steps):
            x = batch.x
            batch.x = self.forward_message.forward(x,edge_attr,edge_index) + self.backward_message.forward(x,edge_attr,edge_index)
        x = self.reduce(batch.x.unsqueeze(1))
        g =  torch.sigmoid(self.gm(x))
        h_v_G = self.fm(x)
        h_G = torch.sum( g * h_v_G , dim = 0 )

        return h_G

class Agent(nn.Module):
    
    def __init__(self,agent_type,node_embedding_dim,M,edge_embedding_dim,hidden_node_dim,graph_dim = 50,prop_steps = 2):
        super(Agent,self).__init__()

        p = node_embedding_dim[0]
        b = node_embedding_dim[1]
        self.M = M # Max num of servers.
        self.agent_type = agent_type

        self.f_G_actionvalue = Graph_Representation(node_embedding_dim,edge_embedding_dim,hidden_node_dim,graph_dim,prop_steps)
        self.f_G_policy = Graph_Representation(node_embedding_dim,edge_embedding_dim,hidden_node_dim,graph_dim,prop_steps)

        if agent_type == "add node":
            '''
            k ∈	[0,M]
            xk ∈ R^(pxb)
            '''
            self.policy_network = nn.ModuleList( [ nn.Sequential(nn.Linear(graph_dim+1,b),nn.Softplus(beta=0.1)) for _ in range(p) ] )
            self.action_value = nn.Linear(graph_dim+1+b*p,1)
        
        elif agent_type == "add edge":
            '''
            k ∈	[1,2^n-1]
            xk ∈ R
            '''
            self.policy_network = nn.Sequential(nn.Linear(graph_dim+1,1),nn.Softplus(beta=0.1))
            self.action_value = nn.Linear(graph_dim+1+1,1)

        elif agent_type == "edit nodes":
            '''
            k ∈	[1,n]
            xk ∈ R^(pxb)
            '''
            self.policy_network = nn.ModuleList( [ nn.Sequential(nn.Linear(graph_dim+1,b),nn.Softplus(beta=0.1)) for _ in range(p) ] )
            self.action_value = nn.Linear(graph_dim+1+b*p,1)
        
        elif agent_type == "edit weights":
            '''
            k ∈	{(i,j)|i<j}
            xk ∈ R
            '''
            self.policy_network = nn.Sequential(nn.Linear(graph_dim+2,1),nn.Sigmoid())
            self.action_value = nn.Linear(graph_dim+2+1,1)
        
        else:
            raise NotSupportedError
        
        self.theta_param = nn.ModuleList( [self.f_G_policy, self.policy_network] )
        self.w_param = nn.ModuleList( [self.f_G_actionvalue,self.action_value] )

    def action(self,batch: Data,done):
        if done:
            n = batch.num_nodes
        else:
            n = batch.num_nodes - 1
        h_G_actionvalue = self.f_G_actionvalue.forward(batch)
        h_G_policy = self.f_G_policy.forward(batch)

        max_Q = torch.tensor(-1.0*np.inf)
        optimal_k = None
        optimal_xk = None
        
        if self.agent_type == "add node":
            '''
            k ∈	[0,M]
            xk ∈ R^(pxb)
            '''
            for k in range(self.M+1):
                k_ = torch.tensor([k]).to(device)
                x = torch.cat([h_G_policy,k_])

                temp = []
                for i in range(self.policy_network.__len__()):
                    temp.append(self.policy_network[i].forward(x).unsqueeze(0))
                xk = torch.cat(temp,dim=0)

                x = torch.cat([h_G_actionvalue,k_,xk.flatten()])
                
                Q = self.action_value.forward(x)
                
                if Q > max_Q:
                    max_Q = Q
                    optimal_k = k
                    optimal_xk = xk

        elif self.agent_type == "add edge":
            '''
            k ∈	[1,2^n-1]
            xk ∈ R
            '''
            for k in range(0,2**n-2):
                k_ = torch.tensor([k]).to(device)
                x = torch.cat([h_G_policy,k_])

                xk = self.policy_network.forward(x)

                x = torch.cat([h_G_actionvalue,k_,xk])

                Q = self.action_value.forward(x)

                if Q > max_Q:
                    max_Q = Q
                    optimal_k = k
                    optimal_xk = xk

        elif self.agent_type == "edit nodes":
            '''
            k ∈	[1,n]
            xk ∈ R^(pxb)
            '''
            if not(done):
                n += 1

            for k in range(n):
                k_ = torch.tensor([k]).to(device)
                x = torch.cat([h_G_policy,k_])

                temp = []
                for i in range(self.policy_network.__len__()):
                    temp.append(self.policy_network[i].forward(x).unsqueeze(0))
                xk = torch.cat(temp,dim=0)

                x = torch.cat([h_G_actionvalue,k_,xk.flatten()])
                
                Q = self.action_value.forward(x)

                if Q > max_Q:
                    max_Q = Q
                    optimal_k = k
                    optimal_xk = xk
        
        elif self.agent_type == "edit weights":
            '''
            k ∈	{(i,j)|i<j}
            xk ∈ R:[0,1]
            '''
            if not(done):
                n += 1

            for i in range(n-1):
                for j in range(i+1,n):
                    k = [i,j]
                    k_ = torch.tensor(k).to(device)
                    x = torch.cat([h_G_policy,k_])

                    xk = self.policy_network.forward(x)

                    x = torch.cat([h_G_actionvalue,k_,xk])

                    Q = self.action_value.forward(x)

                    if Q > max_Q:
                        max_Q = Q
                        optimal_k = k
                        optimal_xk = xk

        else:
            raise NotSupportedError

        return max_Q,optimal_k,optimal_xk
    
    def rn_action(self,batch: Data,k: int):
        h_G_actionvalue = self.f_G_actionvalue.forward(batch)
        h_G_policy = self.f_G_policy.forward(batch)
        
        if self.agent_type == "add node":
            '''
            k ∈	[0,M]
            xk ∈ R^(pxb)
            '''
            k_ = torch.tensor([k]).to(device)
            x = torch.cat([h_G_policy,k_])

            temp = []
            for i in range(self.policy_network.__len__()):
                temp.append(self.policy_network[i].forward(x).unsqueeze(0))
            xk = torch.cat(temp,dim=0)

            x = torch.cat([h_G_actionvalue,k_,xk.flatten()])
            
            Q = self.action_value.forward(x)

        elif self.agent_type == "add edge":
            '''
            k ∈	[1,2^n-1]
            xk ∈ R
            '''
            k_ = torch.tensor([k]).to(device)
            x = torch.cat([h_G_policy,k_])

            xk = self.policy_network.forward(x)

            x = torch.cat([h_G_actionvalue,k_,xk])

            Q = self.action_value.forward(x)

        elif self.agent_type == "edit nodes":
            '''
            k ∈	[1,n]
            xk ∈ R^(pxb)
            '''
            k_ = torch.tensor([k]).to(device)
            x = torch.cat([h_G_policy,k_])

            temp = []
            for i in range(self.policy_network.__len__()):
                temp.append(self.policy_network[i].forward(x).unsqueeze(0))
            xk = torch.cat(temp,dim=0)

            x = torch.cat([h_G_actionvalue,k_,xk.flatten()])
            
            Q = self.action_value.forward(x)

        elif self.agent_type == "edit weights":
            '''
            k ∈	[1,n]
            xk ∈ R
            '''
            k_ = torch.tensor(k).to(device)
            x = torch.cat([h_G_policy,k_])

            xk = self.policy_network.forward(x)

            x = torch.cat([h_G_actionvalue,k_,xk])

            Q = self.action_value.forward(x)

        else:
            raise NotSupportedError

        return xk,Q

    def action_value_calc(self,batch : Data,k,xk):
        h_G_actionvalue = self.f_G_actionvalue.forward(batch)

        if self.agent_type == "add node":
            '''
            k ∈	[0,M]
            xk ∈ R^(pxb)
            '''
            k_ = torch.tensor([k]).to(device)
            x = torch.cat([h_G_actionvalue,k_])

            x = torch.cat([x,xk.flatten()])
            
            Q = self.action_value.forward(x)
   
        elif self.agent_type == "add edge":
            '''
            k ∈	[1,2^n-1]
            xk ∈ R
            '''
            x = torch.cat([h_G_actionvalue,torch.tensor([k]).to(device)])

            x = torch.cat([x,xk])

            Q = self.action_value.forward(x)

        elif self.agent_type == "edit nodes":
            '''
            k ∈	[1,n]
            xk ∈ R^(pxb)
            '''
            x = torch.cat([h_G_actionvalue,torch.tensor([k]).to(device)])

            x = torch.cat([x,xk.flatten()])
            
            Q = self.action_value.forward(x)
   
        elif self.agent_type == "edit weights":
            '''
            k ∈	[1,n]
            xk ∈ R
            '''            
            x = torch.cat([h_G_actionvalue,torch.tensor(k).to(device)])

            x = torch.cat([x,xk])

            Q = self.action_value.forward(x)

        else:
            raise NotSupportedError

        return Q

    def Q_hat(self,batch: Data,done):
        if done:
            n = batch.num_nodes
        else:
            n = batch.num_nodes - 1
        with torch.no_grad():
            h_G_actionvalue = self.f_G_actionvalue.forward(batch)
        h_G_policy = self.f_G_policy.forward(batch)

        Q_sum = torch.zeros((1),device=device,dtype=torch.float64)

        if self.agent_type == "add node":
            '''
            k ∈	[0,M]
            xk ∈ R^(pxb)
            '''
            for k in range(self.M+1):
                k_ = torch.tensor([k]).to(device)
                x = torch.cat([h_G_policy,k_])

                temp = []
                for i in range(self.policy_network.__len__()):
                    temp.append(self.policy_network[i].forward(x).unsqueeze(0))
                xk = torch.cat(temp,dim=0)

                x = torch.cat([h_G_actionvalue,k_,xk.flatten()])
                
                Q_sum += self.action_value.forward(x)
      
        elif self.agent_type == "add edge":
            '''
            k ∈	[1,2^n-1]
            xk ∈ R
            '''
            for k in range(0,2**n-2):
                k_ = torch.tensor([k]).to(device)
                x = torch.cat([h_G_policy,k_])

                xk = self.policy_network.forward(x)

                x = torch.cat([h_G_actionvalue,k_,xk])

                Q_sum += self.action_value.forward(x)

        elif self.agent_type == "edit nodes":
            '''
            k ∈	[1,n]
            xk ∈ R^(pxb)
            '''
            if not(done):
                n += 1
                
            for k in range(n):
                k_ = torch.tensor([k]).to(device)
                x = torch.cat([h_G_policy,k_])

                temp = []
                for i in range(self.policy_network.__len__()):
                    temp.append(self.policy_network[i].forward(x).unsqueeze(0))
                xk = torch.cat(temp,dim=0)

                x = torch.cat([h_G_actionvalue,k_,xk.flatten()])
                
                Q_sum += self.action_value.forward(x)
                
        elif self.agent_type == "edit weights":
            '''
            k ∈	{(i,j)|i<j}
            xk ∈ R
            '''
            if not(done):
                n += 1
            for i in range(n-1):
                for j in range(i+1,n):
                    k = [i,j]
                    k_ = torch.tensor(k).to(device)
                    x = torch.cat([h_G_policy,k_])

                    xk = self.policy_network.forward(x)

                    x = torch.cat([h_G_actionvalue,k_,xk.flatten()])

                    Q_sum += self.action_value.forward(x)

        else:
            raise NotSupportedError

        return Q_sum