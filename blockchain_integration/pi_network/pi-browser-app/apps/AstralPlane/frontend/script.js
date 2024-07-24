// script.js
import Web3 from 'web3';
import { AstralPlaneAsset, AstralPlaneMarketplace, VRController, VRRenderer } from './contracts';

const web3 = new Web3(window.ethereum);

let account;
let astralPlaneAsset;
let astralPlaneMarketplace;
let vrController;
let vrRenderer;

// Connect to Web3
document.getElementById('connect-web3-btn').addEventListener('click', async () => {
  try {
    await window.ethereum.request({ method: 'eth_requestAccounts' });
    account = web3.eth.accounts[0];
    console.log(`Connected to Web3 with account ${account}`);
    document.querySelector('.web3-modal').style.display = 'none';
    initContracts();
  } catch (error) {
    console.error(error);
  }
});

// Initialize contracts
async function initContracts() {
  astralPlaneAsset = new web3.eth.Contract(AstralPlaneAsset.abi, AstralPlaneAsset.address);
  astralPlaneMarketplace = new web3.eth.Contract(AstralPlaneMarketplace.abi, AstralPlaneMarketplace.address);
  vrController = new web3.eth.Contract(VRController.abi, VRController.address);
  vrRenderer = new web3.eth.Contract(VRRenderer.abi, VRRenderer.address);
  console.log('Contracts initialized');
  renderMarketplace();
}

// Render marketplace
async function renderMarketplace() {
  const assets = await astralPlaneMarketplace.methods.getAssets().call();
  const marketplaceGrid = document.querySelector('.marketplace-grid');
  assets.forEach((asset) => {
    const assetCard = document.createElement('div');
    assetCard.className = 'asset-card';
    assetCard.innerHTML = `
      <img src="${asset.image}" alt="${asset.name}">
      <h3>${asset.name}</h3>
      <p>${asset.description}</p>
      <button class="buy-btn">Buy</button>
    `;
    marketplaceGrid.appendChild(assetCard);
  });
}

// Handle buy button clicks
document.addEventListener('click', (event) => {
  if (event.target.classList.contains('buy-btn')) {
    const assetId = event.target.parentNode.dataset.assetId;
    astralPlaneMarketplace.methods.buyAsset(assetId).send({ from: account });
  }
});

// Initialize VR experience
async function initVRExperience() {
  const vrScene = document.querySelector('.vr-container');
  vrRenderer.methods.renderScene().call().then((scene) => {
    vrScene.innerHTML = scene;
  });
}

// Handle VR experience button clicks
document.addEventListener('click', (event) => {
  if (event.target.id === 'vr-experience-btn') {
    initVRExperience();
  }
});
