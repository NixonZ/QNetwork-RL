from torch_geometric.data.data import Data
from agent.agent import Graph_Representation
import torch.nn as nn

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

        # Qtot layers
        self.Qtot_1 = nn.Linear(num_agents,10)
        # for param in self.Qtot_1.parameters():
        #     param.requires_grad = False
        self.Qtot_2 = nn.Linear(10,1)
        # for param in self.Qtot_2.parameters():
        #     param.requires_grad = False
        pass

    def hypernetwork(self):
        for param in self.Qtot_1.parameters():
            param.requires_grad = True
        for param in self.Qtot_2.parameters():
            param.requires_grad = True

    def Q_tot(self,Q_list):
        x = self.Qtot_1.forward(Q_list)
        return self.Qtot_2.forward(x)

    def set_weights(self,batch: Data):
        h_G = self.f_G.forward(batch)

        weights_1 = self.weights_1.forward(h_G)
        weights_1 = weights_1.reshape((10,self.num_agents))
        bias_1 = self.bias_1.forward(h_G)

        self.Qtot_1._parameters['weight'].data = weights_1
        self.Qtot_1._parameters['bias'].data = bias_1

        weights_2 = self.weights_2.forward(h_G)
        weights_2 = weights_2.reshape((1,10))
        bias_2 = self.bias_2.forward(h_G)

        self.Qtot_2._parameters['weight'].data = weights_2
        self.Qtot_1._parameters['bias'].data = bias_2

    def forward(self,batch: Data,Q_list):
        self.set_weights(batch)
        return self.Q_tot(Q_list)
        
