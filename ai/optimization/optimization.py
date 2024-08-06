from .network import NeuralNetworkOptimizer, NeuralNetworkPruningScheduler
from .resource import ResourceOptimizer, ResourceScheduler

def optimize_neural_network(model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer):
    neural_network_optimizer = NeuralNetworkOptimizer(model, criterion, optimizer)
    neural_network_pruning_scheduler = NeuralNetworkPruningScheduler(neural_network_optimizer, 0.2, 5)
    return neural_network_pruning_scheduler

def optimize_resources(model: torch.nn.Module):
    resource_optimizer = ResourceOptimizer(model)
    resource_scheduler = ResourceScheduler(resource_optimizer, 10)
    return resource_scheduler
