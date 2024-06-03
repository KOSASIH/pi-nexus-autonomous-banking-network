# test.py

import web3
from pi_dapp import PIDApp

def test_pi_dapp():
    # Connect to an Ethereum network (e.g., localhost, mainnet, testnet)
    web3 = web3.Web3(web3.HTTPProvider('http://localhost:8545'))

    # Define the contract addresses for each contract
    contract_addresses = {
        'oracle': '0x123...',
        'insurance': '0x456...',
        'nft': '0x789...',
        'exchange': '0xabc...',
        'bank': '0xdef...'
    }

    # Initialize the PI DApp
    pi_dapp = PIDApp(web3, contract_addresses)

    # Test the deposit and withdraw functions
    pi_dapp.deposit('PIC', 100)
    assert pi_dapp.get_balance('PIC') == 100
    pi_dapp.withdraw('PIC', 50)
    assert pi_dapp.get_balance('PIC') == 50

    # Test the create_oracle function
    oracle_id = pi_dapp.create_oracle('example.com', 'example query', 2)
    assert oracle_id == 0

    # Test the create_insurance_product function
    product_id = pi_dapp.create_insurance_product(oracle_id, 100, 200)
    assert product_id == 0

    # Test the purchase_insurance function
    pi_dapp.purchase_insurance(product_id, 100)
    assert pi_dapp.get_balance('PIC') == 0

    # Test the withdraw_insurance_payout function
    pi_dapp.withdraw_insurance_payout(product_id)
    assert pi_dapp.get_balance('PIC') == 1000

    # Test the mint_nft function
    nft_id = pi_dapp.mint_nft(product_id)
    assert nft_id == 0

    # Test the list_asset and fill_order functions
    pi_dapp.list_asset('PIC', 1000)
    order_id = pi_dapp.get_order_book('PIC')[0]['id']
    pi_dapp.fill_order(order_id)
    assert pi_dapp.get_balance('PIC') == 1100

    # Test the set_exchange_parameters and set_bank_parameters functions
    pi_dapp.set_exchange_parameters({'fee': 0.001})
    pi_dapp.set_bank_parameters({'fee': 0.001})
    assert pi_dapp.exchange.get_parameters()['fee'] == 0.001
    assert pi_dapp.bank.get_parameters()['fee'] == 0.001

# Run the tests
test_pi_dapp()
