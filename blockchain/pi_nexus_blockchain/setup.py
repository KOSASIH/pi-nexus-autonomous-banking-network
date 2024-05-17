# setup.py

from setuptools import find_packages, setup

# Project metadata
NAME = "pi_nexus_blockchain"
VERSION = "0.1.0"
AUTHOR = "Your Name"
AUTHOR_EMAIL = "your.email@example.com"
URL = "https://github.com/yourusername/pi_nexus_blockchain"
DESCRIPTION = "A simple blockchain implementation for Pi Nexus."
LONG_DESCRIPTION = open("README.md").read()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

# Package requirements
REQUIRED_PACKAGES = [
    "pytest",
    "cryptography",
]

# Go to the directory where the command was run
HERE = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
