from environment.env import Env
from agent.agent import Agent

class trainer:
    def __init__(self,p: int,b: int,environment: Env, hidden_node_dim: int, graph_dim: int, prop_steps: int,max_nodes: int = 10, buffer_size: int = 100000,lr: int = 0.0001) -> None:
        
        # Agents
        self.agent_addnode = Agent("add node",(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()
        self.agent_addedge = Agent("add edge",(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()
        self.agent_editnode = Agent("edit nodes",(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()
        self.agent_editweights = Agent("edit weights",(p,b),1,hidden_node_dim,graph_dim,prop_steps).double()

        self.env = environment

        self.max_nodes = max_nodes
        self.buffer = []
        self.buffer_size = buffer_size
        self.lr = lr

        pass

    def learn(self):
        pass

    def train(self,epochs):

        done = False
        environment = self.env

        for i in range(epochs):
            temp = []

            # Add node
            data = environment.get_state_torch()
            k,xk = self.agent_addnode.action(data)
            environment.step( ( "add node", (k,xk.detach().numpy()) ) )
            if k == 0:
                done = True
                environment = self.env
            temp.append(
                {
                    "action type" : "add node",
                    "state" : data,
                    "discrete action" : k,
                    "continuous action" : xk
                }
            )

            # Add edge
            data = environment.get_state_torch()
            k,xk = self.agent_addedge.action(data)
            environment.step( ( "add edge", (k,xk.detach().numpy()) ) )
            temp.append(
                {
                    "action type" : "add node",
                    "state" : data,
                    "discrete action" : k,
                    "continuous action" : xk
                }
            )

            # Edit node

            # Edit weights

            if len(self.buffer) > self.buffer_size:
                self.learn()
                self.buffer = []

            if environment.n > self.max_nodes:
                done = True
                environment = self.env

            self.buffer.append(
                {
                    "Epoch" : i,
                    "step" : temp,
                    "done" : done
                }
            )



            
