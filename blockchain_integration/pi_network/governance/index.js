import { Web3 } from "web3";
import { ethers } from "ethers";
import { Governance } from "./Governance";
import { GovernanceController } from "./GovernanceController";

const web3 = new Web3(
  new Web3.providers.HttpProvider(
    "https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
  ),
);
const wallet = new ethers.Wallet("0x1234567890abcdef", web3);

const governance = new Governance(web3, wallet);
const governanceController = new GovernanceController(governance);

export default governanceController;
