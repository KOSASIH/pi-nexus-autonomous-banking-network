# Tokenization Service

This directory contains the implementation of a smart contract for tokenizing assets and a Python script for managing tokenized assets.

## Directory Structure

- `asset_token.sol`: Smart contract for managing tokenized assets.
- `tokenization_service.py`: Python script for interacting with the tokenization contract.
- `README.md`: Documentation for the tokenization service.

## Asset Token Contract (`asset_token.sol`)

The Asset Token contract allows users to create assets, mint tokens for those assets, and burn tokens as needed. It tracks asset details and emits events for asset creation, token minting, and token burning.

### Functions

- `createAsset(string memory _name, string memory _description, uint256 _value)`: Creates a new asset.
- `mintTokens(uint256 _assetId, address _to, uint256 _amount)`: Mints tokens for a specific asset.
- `burnTokens(uint256 _assetId, uint256 _amount)`: Burns tokens for a specific asset.
- `getAssetDetails(uint256 _assetId)`: Retrieves the details of a specific asset.

### Events

- `AssetCreated(uint256 indexed assetId, string name, string description, uint256 value)`: Emitted when a new asset is created.
- `TokensMinted(uint256 indexed assetId, address indexed to, uint256 amount)`: Emitted when tokens are minted for an asset.
- `TokensBurned(uint256 indexed assetId, address indexed from, uint256 amount)`: Emitted when tokens are burned for an asset.

## Tokenization Service (`tokenization_service.py`)

The Tokenization Service is a Python application that interacts with the Asset Token contract. It allows users to create assets, mint tokens, burn tokens, and retrieve asset details.

### Installation

1. Install the required packages:
   ```bash
   1 pip install web3
   ```

## Usage

1. **Initialize the Tokenization Service**: Update the provider_url, contract_address, and abi in the tokenization_service.py file.

2. **Create an Asset**: Call the create_asset method with the owner's account, private key, asset name, description, and value.

3. **Mint Tokens**: Call the mint_tokens method with the owner's account, private key, asset ID, recipient address, and amount of tokens to mint.

4. **Burn Tokens**: Call the burn_tokens method with the owner's account, private key, asset ID, and amount of tokens to burn.

5. **Get Asset Details**: Call the get_asset_details method with the asset ID to retrieve its details.

## License
This project is licensed under the MIT License.
