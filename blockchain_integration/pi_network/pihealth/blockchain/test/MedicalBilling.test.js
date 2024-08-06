const chai = require('chai');
const chaiAsPromised = require('chai-as-promised');
const Web3 = require('web3');
const TruffleContract = require('truffle-contract');
const MedicalBilling = require('../build/contracts/MedicalBilling.json');

chai.use(chaiAsPromised);

const web3 = new Web3(new Web3.providers.HttpProvider('http://localhost:8545'));

const medicalBillingContract = TruffleContract(MedicalBilling);
medicalBillingContract.setProvider(web3.currentProvider);

describe('MedicalBilling', () => {
  let accounts;
  let medicalBillingInstance;

  beforeEach(async () => {
    accounts = await web3.eth.getAccounts();
    medicalBillingInstance = await medicalBillingContract.new({
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });
  });

  it('should create a new medical billing record', async () => {
    const patientName = 'John Doe';
    const medicalProcedure = 'Surgery';
    const cost = web3.utils.toWei('100', 'ether');

    const result = await medicalBillingInstance.createMedicalBillingRecord(patientName, medicalProcedure, cost, {
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    expect(result.receipt.status).to.equal(true);
  });

    it('should get a medical billing record', async () => {
    const patientName = 'John Doe';
    const medicalProcedure = 'Surgery';
    const cost = web3.utils.toWei('100', 'ether');

    await medicalBillingInstance.createMedicalBillingRecord(patientName, medicalProcedure, cost, {
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    const result = await medicalBillingInstance.getMedicalBillingRecord(accounts[0], {
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    expect(result).to.equal(patientName);
  });

  it('should update a medical billing record', async () => {
    const patientName = 'John Doe';
    const medicalProcedure = 'Surgery';
    const cost = web3.utils.toWei('100', 'ether');

    await medicalBillingInstance.createMedicalBillingRecord(patientName, medicalProcedure, cost, {
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    const newPatientName = 'Jane Doe';
    const newMedicalProcedure = 'Checkup';
    const newCost = web3.utils.toWei('50', 'ether');

    const result = await medicalBillingInstance.updateMedicalBillingRecord(newPatientName, newMedicalProcedure, newCost, {
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    expect(result.receipt.status).to.equal(true);
  });

  it('should delete a medical billing record', async () => {
    const patientName = 'John Doe';
    const medicalProcedure = 'Surgery';
    const cost = web3.utils.toWei('100', 'ether');

    await medicalBillingInstance.createMedicalBillingRecord(patientName, medicalProcedure, cost, {
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    const result = await medicalBillingInstance.deleteMedicalBillingRecord(accounts[0], {
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    expect(result.receipt.status).to.equal(true);
  });
});
