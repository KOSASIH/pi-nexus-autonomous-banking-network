import os
import socket
import psutil
import torch
from abc import ABC, abstractmethod

class Node(ABC):
    def __init__(self, node_id, node_type, gpu_id=0):
        self.node_id = node_id
        self.node_type = node_type
        self.gpu_id = gpu_id
        self.gpu = torch.cuda.device(gpu_id)
        self.cpu_util = 0
        self.memory_util = 0
        self.gpu_util = 0
        self.gpu_memory_util = 0

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def get_node_info(self):
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'gpu_id': self.gpu_id,
            'cpu_util': self.cpu_util,
            'memory_util': self.memory_util,
            'gpu_util': self.gpu_util,
            'gpu_memory_util': self.gpu_memory_util
        }

    def update_utilization(self):
        self.cpu_util = psutil.cpu_percent()
        self.memory_util = psutil.virtual_memory().percent
        gpu = torch.cuda.device(self.gpu_id)
        self.gpu_util = gpu.utilization
        self.gpu_memory_util = gpu.memory_utilization

class WorkerNode(Node):
    def __init__(self, node_id, gpu_id=0):
        super().__init__(node_id, 'worker', gpu_id)
        self.tasks = []

    def start(self):
        print(f'Starting worker node {self.node_id}...')

    def stop(self):
        print(f'Stopping worker node {self.node_id}...')

    def add_task(self, task):
        self.tasks.append(task)

    def run_task(self, task):
        print(f'Running task {task} on worker node {self.node_id}...')
        # Run the task on the GPU
        torch.cuda.set_device(self.gpu_id)
        # ...

class ParameterServerNode(Node):
    def __init__(self, node_id, gpu_id=0):
        super().__init__(node_id, 'parameter_server', gpu_id)
        self.parameters = {}

    def start(self):
        print(f'Starting parameter server node {self.node_id}...')

    def stop(self):
        print(f'Stopping parameter server node {self.node_id}...')

    def update_parameters(self, parameters):
        self.parameters.update(parameters)

    def get_parameters(self):
        return self.parameters
