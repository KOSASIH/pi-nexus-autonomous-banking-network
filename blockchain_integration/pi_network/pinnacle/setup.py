import os
import sys
from setuptools import setup, find_packages

# Define project metadata
PROJECT_NAME = "pinnacle"
VERSION = "0.1.0"
AUTHOR = "KOSASIH"
EMAIL = "kosasihg88@gmail.com"

# Define dependencies
DEPENDENCIES = [
    "web3==5.23.1",
    "eth-sig-util==1.3.2",
    "py-solc-x==1.3.3",
    "scikit-learn==1.0.2",
    "tensorflow==2.6.0",
    "numpy==1.21.2",
    "pandas==1.3.5",
    "matplotlib==3.5.1",
    "seaborn==0.11.2",
]

# Define project packages
PACKAGES = find_packages()

# Define project scripts
SCRIPTS = []

# Set up the project
setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description="Decentralized, AI-Powered, and Interoperable Cross-Chain Liquidity Aggregator for Pi Network",
    long_description=open("README.md").read(),
    url="https://github.com/KOSASIH/pi-nexus-autonomous-banking-network/tree/main/blockchain_integration/pi_network/pinnacle",
    packages=PACKAGES,
    scripts=SCRIPTS,
    install_requires=DEPENDENCIES,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
