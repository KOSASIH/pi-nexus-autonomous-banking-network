import os
import subprocess

class NodeSoftware:
    def __init__(self, software_id, version, dependencies):
        self.software_id = software_id
        self.version = version
        self.dependencies = dependencies

    def start(self):
        self.install_dependencies()
        self.run_software()

    def stop(self):
        self.stop_software()

    def install_dependencies(self):
        for dependency in self.dependencies:
            subprocess.run(['pip', 'install', dependency])

    def run_software(self):
        os.system(f'python -m {self.software_id}=={self.version}')

    def stop_software(self):
        os.system(f'pkill -f {self.software_id}')
