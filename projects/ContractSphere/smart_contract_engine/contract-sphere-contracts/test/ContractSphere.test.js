const ContractSphere = artifacts.require("ContractSphere");

contract("ContractSphere", (accounts) => {
  let contractSphere;

  beforeEach(async () => {
    contractSphere = await ContractSphere.deployed();
  });

  it("should create a new contract", async () => {
    const metadata = "This is a test contract";
    await contractSphere.createContract(metadata);
    const contractId = await contractSphere.getContractId(metadata);
    assert(contractId > 0, "Contract ID should be greater than 0");
  });

  it("should transfer a contract", async () => {
    const metadata = "This is a test contract";
    await contractSphere.createContract(metadata);
    const contractId = await contractSphere.getContractId(metadata);
    await contractSphere.transferContract(contractId, accounts[1]);
    const newOwner = await contractSphere.getContractOwner(contractId);
    assert.equal(newOwner, accounts[1], "Contract owner should be updated");
  });

  it("should update a contract", async () => {
    const metadata = "This is a test contract";
    await contractSphere.createContract(metadata);
    const contractId = await contractSphere.getContractId(metadata);
    await contractSphere.updateContract(contractId, "This is an updated contract");
    const updatedMetadata = await contractSphere.getContractMetadata(contractId);
    assert.equal(updatedMetadata, "This is an updated contract", "Contract metadata should be updated");
  });
});
