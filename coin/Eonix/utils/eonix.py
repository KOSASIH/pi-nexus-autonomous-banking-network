import os
import json
from config import config
from blockchain import Blockchain
from wallet import Wallet
from contracts import EonixContract
from database import EonixDatabase
from ai import EonixAI
from ml import EonixML
from nlp import EonixNLP
from qc import EonixQC
from ar import EonixAR
from agi import EonixAGI
from advanced_ai import EonixAdvancedAI
from cybersecurity import EonixCybersecurity
from iot import EonixIoT

class Eonix:
    def __init__(self):
        self.config = config
        self.blockchain = Blockchain(self.config["blockchain"])
        self.wallet = Wallet()
        self.contract_manager = EonixContractManager()
        self.database = EonixDatabase(self.config["database"])
        self.ai = EonixAI(self.config["ai"])
        self.ml = EonixML()
        self.nlp = EonixNLP()
        self.qc = EonixQC()
        self.ar = EonixAR()
        self.agi = EonixAGI()
        self.advanced_ai = EonixAdvancedAI()
        self.cybersecurity = EonixCybersecurity(self.config["cybersecurity"])
        self.iot = EonixIoT(self.config["iot"])

    # ... (rest of the code)
