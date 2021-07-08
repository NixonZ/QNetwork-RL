import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.utils import from_networkx
from torch_geometric.data import Data,DataLoader

from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing

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

        self.reduce = nn.Sequential(
            nn.Conv2d(2,10,(node_embedding_dim[0],1),padding=(node_embedding_dim[0],0)),
            nn.Conv2d(10,100,(node_embedding_dim[0],node_embedding_dim[1]//2+1)),
            nn.Conv2d(100,50,(node_embedding_dim[0],node_embedding_dim[1]//4),stride=(1,node_embedding_dim[1]//16)),
            nn.Flatten(start_dim=1),
            nn.Linear(50*3*5, hidden_node_dim)
        )

        self.node_data = nn.ModuleList( [ nn.Linear(hidden_node_dim + edge_embedding_dim,node_embedding_dim[1]) for _ in range(node_embedding_dim[0]) ] )

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
        return torch.cat(temp,dim=1)
    

class Graph_Representation(nn.Module):

    def __init__(self,node_embedding_dim,edge_embedding_dim,hidden_node_dim,graph_dim = 50,prop_steps = 2):
        super(Graph_Representation,self).__init__()

        # Message Passing Layers
        self.prop_steps = prop_steps
        self.forward_message = MPNN(node_embedding_dim,edge_embedding_dim,hidden_node_dim,mode='forward')
        self.backward_message = MPNN(node_embedding_dim,edge_embedding_dim,hidden_node_dim,mode='backward')
        # self.MPN_list = nn.ModuleList( [ MPNN(node_embedding_dim,edge_embedding_dim,hidden_node_dim) for _ in range(prop_steps) ] )

        # Node Dimensionality reduction
        self.reduce = nn.Sequential(
            nn.Conv2d(1,10,(node_embedding_dim[0],1),padding=(node_embedding_dim[0],0)),
            nn.Conv2d(10,100,(node_embedding_dim[0],node_embedding_dim[1]//2+1)),
            nn.Conv2d(100,50,(node_embedding_dim[0],node_embedding_dim[1]//4),stride=(1,node_embedding_dim[1]//16)),
            nn.Flatten(start_dim=1),
            nn.Linear(50*3*5, hidden_node_dim)
        )

        # Learning Graph representation Layers
        self.gm = nn.Linear(hidden_node_dim,graph_dim)
        self.fm = nn.Linear(hidden_node_dim,graph_dim)

    def forward(self,batch):
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