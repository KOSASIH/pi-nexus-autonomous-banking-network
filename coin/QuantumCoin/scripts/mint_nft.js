// mint_nft.js
const { ethers } = require("ethers");
const NFTContractABI = require("./artifacts/NFTContract.json"); // Adjust the path as necessary

const provider = new ethers.providers.JsonRpcProvider("YOUR_INFURA_OR_ALCHEMY_URL");
const wallet = new ethers.Wallet("YOUR_PRIVATE_KEY", provider);
const nftContractAddress = "YOUR_NFT_CONTRACT_ADDRESS";

async function mintNFT(recipient, tokenURI) {
    const nftContract = new ethers.Contract(nftContractAddress, NFTContractABI.abi, wallet);

    try {
        const tx = await nftContract.mintNFT(recipient, tokenURI);
        console.log(`Minting NFT... Transaction Hash: ${tx.hash}`);
        await tx.wait();
        console.log("NFT minted successfully!");
    } catch (error) {
        console.error("Error minting NFT:", error);
    }
}

// Example usage
const recipientAddress = "RECIPIENT_ADDRESS"; // Replace with the recipient's address
const metadataURI = "https://example.com/metadata.json"; // Replace with the actual metadata URI

mintNFT(recipientAddress, metadataURI);
