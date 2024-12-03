// identityVerification.js
const { ethers } = require("ethers");

async function verifyIdentity(userAddress) {
    const identityContract = new ethers.Contract(identityContractAddress, identityABI, provider);
    const isVerified = await identityContract.isVerified(userAddress);
    return isVerified;
}
