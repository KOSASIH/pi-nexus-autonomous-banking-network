# interoperability_tests.py
# A Python file to write tests for cross-chain interoperability
import pytest
from hyperbridge import Hyperbridge
from hyperbridge.chain import Chain

def test_transfer_assets():
    # Initialize the bridge with a configuration
    config = Config()
    bridge = Hyperbridge(config)

    # Register two chains with the bridge
    chain1 = Chain("Chain1")
    chain2 = Chain("Chain2")
    bridge.register_chain(chain1)
    bridge.register_chain(chain2)

    # Transfer assets from chain1 to chain2
    amount = 100
    bridge.transfer_assets(chain1, chain2, amount)

    # Verify that the assets were transferred successfully
    assert chain2.balance == amount

if __name__ == "__main__":
    pytest.main([__file__])
