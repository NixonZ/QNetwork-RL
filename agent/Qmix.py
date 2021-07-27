import torch
from torch_geometric.data.data import Data
from agent.agent import Graph_Representation
import torch.nn as nn
import torch.nn.functional as F

class Qmix(nn.Module):

    def __init__(self,num_agents,node_embedding_dim,edge_embedding_dim,hidden_node_dim,graph_dim,prop_steps) -> None:
        super(Qmix,self).__init__()
        self.f_G = Graph_Representation(node_embedding_dim,edge_embedding_dim,hidden_node_dim,graph_dim,prop_steps)
        self.num_agents = num_agents
        #hypernetwork layers
        self.weights_1 = nn.Linear(graph_dim,num_agents*10)
        self.bias_1 = nn.Linear(graph_dim,10)
        
        self.weights_2 = nn.Linear(graph_dim,10*1)
        self.bias_2 = nn.Linear(graph_dim,1)
        pass
    
    def set_weights(self,batch: Data):
        h_G = self.f_G.forward(batch)

        w1 = self.weights_1.forward(h_G).reshape((10,self.num_agents))
        b1 = self.bias_1.forward(h_G).reshape((10))

        w2 = self.weights_2.forward(h_G).reshape((1,10))
        b2 = self.bias_2.forward(h_G).reshape((1))

        return w1,b1,w2,b2

    def forward(self,batch: Data,Q_list,hat=False):
        if not(hat):
            w1,b1,w2,b2 = self.set_weights(batch)
        else:
            with torch.no_grad():
                w1,b1,w2,b2 = self.set_weights(batch)
                
        z1 = torch.matmul(w1,Q_list)  + b1
        z2 = F.elu(z1)
        z = torch.matmul(w2, z2) + b2

        return z