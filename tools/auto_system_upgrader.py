import os
import sys
import subprocess
from setuptools import find_packages

# Define the list of packages to upgrade
PACKAGES = [
    'package1',
    'package2',
    'package3',
    # Add more packages as needed
]

# Define a function to upgrade a package
def upgrade_package(package):
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', package])
    except subprocess.CalledProcessError as e:
        print(f'Error upgrading package {package}: {e}')

# Define a function to upgrade all packages
def upgrade_all_packages():
    for package in PACKAGES:
        upgrade_package(package)

# Define a function to install all dependencies
def install_dependencies():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    except subprocess.CalledProcessError as e:
        print(f'Error installing dependencies: {e}')

# Define a function to upgrade the system
def upgrade_system():
    # Upgrade all packages
    upgrade_all_packages()

    # Install all dependencies
    install_dependencies()

# Example usage
if __name__ == '__main__':
    upgrade_system()
