const chai = require('chai');
const chaiAsPromised = require('chai-as-promised');
const Web3 = require('web3');
const TruffleContract = require('truffle-contract');
const HealthRecord = require('../build/contracts/HealthRecord.json');

chai.use(chaiAsPromised);

const web3 = new Web3(new Web3.providers.HttpProvider('http://localhost:8545'));

const healthRecordContract = TruffleContract(HealthRecord);
healthRecordContract.setProvider(web3.currentProvider);

describe('HealthRecord', () => {
  let accounts;
  let healthRecordInstance;

  beforeEach(async () => {
    accounts = await web3.eth.getAccounts();
    healthRecordInstance = await healthRecordContract.new({
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });
  });

  it('should create a new health record', async () => {
    const patientName = 'John Doe';
    const patientAddress = '123 Main St';
    const patientPhone = '1234567890';

    const result = await healthRecordInstance.createHealthRecord(patientName, patientAddress, patientPhone, {
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    expect(result.receipt.status).to.equal(true);
  });

  it('should get a health record', async () => {
    const patientName = 'John Doe';
    const patientAddress = '123 Main St';
    const patientPhone = '1234567890';

    await healthRecordInstance.createHealthRecord(patientName, patientAddress, patientPhone, {
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    const result = await healthRecordInstance.getHealthRecord(accounts[0], {
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    expect(result).to.equal(patientName);
  });

  it('should update a health record', async () => {
    const patientName = 'John Doe';
    const patientAddress = '123 Main St';
    const patientPhone = '1234567890';

    await healthRecordInstance.createHealthRecord(patientName, patientAddress, patientPhone, {
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    const newPatientName = 'Jane Doe';
    const newPatientAddress = '456 Elm St';
    const newPatientPhone = '9876543210';

    const result = await healthRecordInstance.updateHealthRecord(newPatientName, newPatientAddress, newPatientPhone, {
      from: accounts[0],
      gas: 5000000,
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    expect(result.receipt.status).to.equal(true);
  });
});
