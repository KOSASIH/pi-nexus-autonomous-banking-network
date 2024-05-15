# hyperledger.py
import os
import sys
from fabric_sdk_py import FabricSDK

class HyperledgerFabric:
    def __init__(self, network, user, org, peer):
        self.sdk = FabricSDK({
            'network': network,
            'user': user,
            'org': org,
            'peer': peer
        })

    def query_ledger(self, chaincode_id, fcn, args):
        try:
            tx_proposal = self.sdk.new_transaction_proposal()
            tx_proposal.chaincode_id = chaincode_id
            tx_proposal.fcn = fcn
            tx_proposal.args = args

            response = self.sdk.send_transaction_proposal(tx_proposal)
            if response.status == 200:
                return response.result
            else:
                raise Exception(f"Error querying ledger: {response.status}")
        except Exception as e:
            print(f"Error: {e}")

    def invoke_chaincode(self, chaincode_id, fcn, args):
        try:
            tx_proposal = self.sdk.new_transaction_proposal()
            tx_proposal.chaincode_id = chaincode_id
            tx_proposal.fcn = fcn
            tx_proposal.args = args

            response = self.sdk.send_transaction_proposal(tx_proposal)
            if response.status == 200:
                return response.result
            else:
                raise Exception(f"Error invoking chaincode: {response.status}")
        except Exception as e:
            print(f"Error: {e}")

    def deploy_chaincode(self, chaincode_path, chaincode_id, chaincode_version):
        try:
            deploy_proposal = self.sdk.new_chaincode_deployment_proposal()
            deploy_proposal.chaincode_path = chaincode_path
            deploy_proposal.chaincode_id = chaincode_id
            deploy_proposal.chaincode_version = chaincode_version

            response = self.sdk.send_chaincode_deployment_proposal(deploy_proposal)
            if response.status == 200:
                return response.result
            else:
                raise Exception(f"Error deploying chaincode: {response.status}")
        except Exception as e:
            print(f"Error: {e}")
