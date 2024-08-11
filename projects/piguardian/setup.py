from setuptools import setup, find_packages

setup(
    name='PiGuardian',
    version='1.0.0',
    description='A Decentralized, AI-Powered, Real-Time Network Integrity and Security Monitoring System',
    author='KOSASIH',
    author_email='kosasihg88@gmail.com',
    url='https://github.com/KOSASIH/pi-nexus-autonomous-banking-network/tree/main/projects/piguardian',
    packages=find_packages(),
    install_requires=[
        'requests==2.25.1',
        'statsmodels==0.12.2',
        'plotly==4.14.3',
        'numpy==1.22.0',
        'pandas==1.3.1',
        'scikit-learn==0.24.2',
        'matplotlib==3.4.3',
        'seaborn==0.11.2',
        'dht-sensor-library==1.2.3',
        'sqlite3==2.6.0'
    ],
    tests_require=[
        'pytest==6.2.4',
        'pytest-cov==2.12.1'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Security',
        'Topic :: System :: Networking :: Monitoring'
    ],
    keywords='piguardian decentralized ai-powered real-time network integrity security monitoring'
)
