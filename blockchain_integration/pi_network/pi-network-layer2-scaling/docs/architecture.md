# Architecture Overview

## Introduction
This document outlines the architecture of the Pi Network Layer 2 Scaling project. The architecture is designed to enhance scalability and usability through Layer 2 solutions.

## Components
- **Main Chain**: The primary blockchain where the Pi Network operates.
- **Layer 2 Solutions**: Off-chain solutions that process transactions to improve speed and reduce costs.
  - **Optimistic Rollups**: A method that allows transactions to be processed off-chain while ensuring security through fraud proofs.
  - **zk-Rollups**: Utilizes zero-knowledge proofs to bundle multiple transactions into a single proof, enhancing privacy and scalability.
  - **State Channels**: A mechanism that allows participants to transact off-chain and only settle on-chain when necessary.

## Interaction Flow
1. Users initiate transactions on the Layer 2 solution.
2. Transactions are processed off-chain.
3. Final states are periodically submitted to the main chain for settlement.

## Conclusion
The architecture aims to provide a scalable and efficient solution for the Pi Network, enabling a seamless user experience.
