import pip
import pkg_resources

class DependencyUpdater:
    def __init__(self):
        self.dependencies = []

    def add_dependency(self, package):
        self.dependencies.append(package)

    def update_dependencies(self):
        for package in self.dependencies:
            try:
                pkg_resources.get_distribution(package)
            except pkg_resources.DistributionNotFound:
                print(f"Installing {package}...")
                pip.main(['install', package])

# Example usage:
updater = DependencyUpdater()
updater.add_dependency("cryptography")
updater.add_dependency("requests")
updater.update_dependencies()
