const { ethers } = require("hardhat");
const { ChainlinkOracle } = require("../contracts/oracles/ChainlinkOracle.sol");
const { RegulatoryKnowledgeGraph } = require("../contracts/RegulatoryKnowledgeGraph.sol");

describe("Chainlink Oracle and Regulatory Knowledge Graph", () => {
  let chainlinkOracle;
  let regulatoryKnowledgeGraph;

  beforeEach(async () => {
    // Deploy the contracts
    const chainlinkOracleFactory = new ethers.ContractFactory(ChainlinkOracle.abi, ChainlinkOracle.bytecode);
    chainlinkOracle = await chainlinkOracleFactory.deploy("0xLINK_TOKEN_ADDRESS", "0xORACLE_ADDRESS");
    await chainlinkOracle.deployed();

    const regulatoryKnowledgeGraphFactory = new ethers.ContractFactory(RegulatoryKnowledgeGraph.abi, RegulatoryKnowledgeGraph.bytecode);
    regulatoryKnowledgeGraph = await regulatoryKnowledgeGraphFactory.deploy();
    await regulatoryKnowledgeGraph.deployed();

    // Set up the Chainlink Oracle as the data provider for the Regulatory Knowledge Graph
    await regulatoryKnowledgeGraph.setChainlinkOracle(chainlinkOracle.address);
  });

  it("should request data from the Chainlink Oracle", async () => {
    // Request data from the Chainlink Oracle
    const requestId = await chainlinkOracle.requestData("latestPrice");
    expect(requestId).to.be.a("string");

    // Fulfill the data request
    await chainlinkOracle.fulfillDataRequest(requestId, "100.0");
    expect(await regulatoryKnowledgeGraph.getLatestPrice()).to.equal("100.0");
  });

  it("should update the regulatory compliance status", async () => {
    // Update the regulatory compliance status
    await regulatoryKnowledgeGraph.updateRegulatoryCompliance("latestPrice", true);
    expect(await regulatoryKnowledgeGraph.getRegulatoryCompliance("latestPrice")).to.equal(true);
  });
});
