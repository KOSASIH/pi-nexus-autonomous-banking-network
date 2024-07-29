# Pi Network Smart Contract Guide

Welcome to the Pi Network Smart Contract Guide! This guide provides detailed information on how to create, deploy, and interact with smart contracts on the Pi Network.

## Creating a Smart Contract

To create a smart contract on the Pi Network, you'll need to write a contract in a programming language such as Solidity or Vyper. You can then compile the contract using a compiler such as `solc` or `vyper`.

## Deploying a Smart Contract

To deploy a smart contract on the Pi Network, you'll need to use the `pi-cli` command-line tool. You can deploy a contract by running the following command:

`pi-cli deploy --contract <contract_file> --gas <gas_limit>`

Replace `<contract_file>` with the path to your compiled contract file, and `<gas_limit>` with the maximum amount of gas you're willing to spend on the deployment.

## Interacting with a Smart Contract

Once a smart contract is deployed, you can interact with it using the `pi-cli` command-line tool. You can call a function on a contract by running the following command:

`pi-cli call --contract <contract_address> --function <function_name> --args <args>`

Replace `<contract_address>` with the address of the contract, `<function_name>` with the name of the function to call, and `<args>` with the arguments to pass to the function.

## Smart Contract Examples

* **Hello World**: A simple smart contract that returns a "Hello, World!" message.
* **Token Contract**: A smart contract that implements a token economy.
* **Auction Contract**: A smart contract that implements an auction mechanism.

## Next Steps

* Learn more about the Pi Network's architecture and features in our [Getting Started](getting_started.md) guide.
* Explore the Pi Network's API in our [API Reference](api_reference.md).
