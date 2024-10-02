# deployment_cd/deploy_engine.py
import subprocess


class DeployEngine:
    def __init__(self):
        self.kubernetes = Kubernetes()

    def deploy(self):
        self.kubernetes.apply_configurations()

    def rollback(self):
        self.kubernetes.rollback_configurations()
