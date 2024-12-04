// test_nft.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("NFTContract", function () {
    let NFTContract;
    let nft;
    let owner;
    let addr1;
    let addr2;

    beforeEach(async function () {
        NFTContract = await ethers.getContractFactory("NFTContract");
        [owner, addr1, addr2] = await ethers.getSigners();
        nft = await NFTContract.deploy();
        await nft.deployed();
    });

    it("Should mint an NFT and set the correct token URI", async function () {
        const tokenURI = "https://example.com/metadata/1.json";
        await nft.mintNFT(addr1.address, tokenURI);

        const tokenId = 0; // First token ID
        expect(await nft.ownerOf(tokenId)).to.equal(addr1.address);
        expect(await nft.tokenURI(tokenId)).to.equal(tokenURI);
    });

    it("Should burn an NFT", async function () {
        const tokenURI = "https://example.com/metadata/1.json";
        await nft.mintNFT(addr1.address, tokenURI);
        const tokenId = 0; // First token ID

        await nft.connect(addr1).burn(tokenId);
        await expect(nft.ownerOf(tokenId)).to.be.revertedWith("ERC721: owner query for nonexistent token");
    });

    it("Should not allow non-owners to burn an NFT", async function () {
        const tokenURI = "https://example.com/metadata/1.json";
        await nft.mintNFT(addr1.address, tokenURI);
        const tokenId = 0; // First token ID

        await expect(nft.connect(addr2).burn(tokenId)).to.be.revertedWith("You are not the owner of this token");
    });
});
