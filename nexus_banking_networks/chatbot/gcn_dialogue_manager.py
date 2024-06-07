import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCNDialogueManager(nn.Module):
    def __init__(self, num_nodes, num_edges, num_actions):
        super(GCNDialogueManager, self).__init__()
        self.gcn_conv = GCNConv(num_nodes, num_edges)
        self.action_predictor = nn.Linear(num_nodes, num_actions)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn_conv(x, edge_index)
        action_logits = self.action_predictor(x)
        return action_logits

# Example usage
dialogue_manager = GCNDialogueManager(num_nodes=10, num_edges=20, num_actions=5)
data = Data(x=torch.tensor([[1, 2, 3], [4, 5, 6]]), edge_index=torch.tensor([[0, 1], [1, 2], [2, 0]]))
action_logits = dialogue_manager(data)
print(f'Action logits: {action_logits}')
