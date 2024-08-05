const Web3 = require('web3');
const contract = require('truffle-contract');
const UserContract = contract(require('./UserContract.json'));

class UserController {
  constructor() {
    this.web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
    this.userContract = UserContract.at('0x...YOUR_CONTRACT_ADDRESS...');
  }

  async createUser(name, email) {
    try {
      const txCount = await this.web3.eth.getTransactionCount();
      const tx = {
        from: '0x...YOUR_ACCOUNT_ADDRESS...',
        to: this.userContract.address,
        value: '0',
        gas: '200000',
        gasPrice: '20',
        nonce: txCount,
        data: this.userContract.createUser.getData(name, email)
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...YOUR_PRIVATE_KEY...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async getUserData(address) {
    try {
      const userData = await this.userContract.getUserData(address);
      return {
        name: userData[0],
        email: userData[1],
        balance: userData[2].toNumber(),
        ownedCourses: userData[3].map(id => id.toNumber()),
        enrolledCourses: userData[4].map(id => id.toNumber()),
        createdCourses: userData[5].map(id => id.toNumber())
      };
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async updateUser(address, name, email) {
    try {
      const txCount = await this.web3.eth.getTransactionCount();
      const tx = {
        from: '0x...YOUR_ACCOUNT_ADDRESS...',
        to: this.userContract.address,
        value: '0',
        gas: '200000',
        gasPrice: '20',
        nonce: txCount,
        data: this.userContract.updateUser.getData(address, name, email)
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...YOUR_PRIVATE_KEY...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async deleteUser(address) {
    try {
      const txCount = await this.web3.eth.getTransactionCount();
      const tx = {
        from: '0x...YOUR_ACCOUNT_ADDRESS...',
        to: this.userContract.address,
        value: '0',
        gas: '200000',
        gasPrice: '20',
        nonce: txCount,
        data: this.userContract.deleteUser.getData(address)
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...YOUR_PRIVATE_KEY...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async addOwnedCourse(address, courseId) {
    try {
      const txCount = await this.web3.eth.getTransactionCount();
      const tx = {
        from: '0x...YOUR_ACCOUNT_ADDRESS...',
        to: this.userContract.address,
        value: '0',
        gas: '200000',
        gasPrice: '20',
        nonce: txCount,
        data: this.userContract.addOwnedCourse.getData(address, courseId)
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...YOUR_PRIVATE_KEY...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async addEnrolledCourse(address, courseId) {
    try {
      const txCount = await this.web3.eth.getTransactionCount();
      const tx = {
        from: '0x...YOUR_ACCOUNT_ADDRESS...',
        to: this.userContract.address,
        value: '0',
        gas: '200000',
        gasPrice: '20',
        nonce: txCount,
        data: this.userContract.addEnrolledCourse.getData(address, courseId)
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...YOUR_PRIVATE_KEY...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

    async addCreatedCourse(address, courseId) {
    try {
      const txCount = await this.web3.eth.getTransactionCount();
      const tx = {
        from: '0x...YOUR_ACCOUNT_ADDRESS...',
        to: this.userContract.address,
        value: '0',
        gas: '200000',
        gasPrice: '20',
        nonce: txCount,
        data: this.userContract.addCreatedCourse.getData(address, courseId)
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...YOUR_PRIVATE_KEY...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async removeOwnedCourse(address, courseId) {
    try {
      const txCount = await this.web3.eth.getTransactionCount();
      const tx = {
        from: '0x...YOUR_ACCOUNT_ADDRESS...',
        to: this.userContract.address,
        value: '0',
        gas: '200000',
        gasPrice: '20',
        nonce: txCount,
        data: this.userContract.removeOwnedCourse.getData(address, courseId)
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...YOUR_PRIVATE_KEY...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async removeEnrolledCourse(address, courseId) {
    try {
      const txCount = await this.web3.eth.getTransactionCount();
      const tx = {
        from: '0x...YOUR_ACCOUNT_ADDRESS...',
        to: this.userContract.address,
        value: '0',
        gas: '200000',
        gasPrice: '20',
        nonce: txCount,
        data: this.userContract.removeEnrolledCourse.getData(address, courseId)
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...YOUR_PRIVATE_KEY...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async removeCreatedCourse(address, courseId) {
    try {
      const txCount = await this.web3.eth.getTransactionCount();
      const tx = {
        from: '0x...YOUR_ACCOUNT_ADDRESS...',
        to: this.userContract.address,
        value: '0',
        gas: '200000',
        gasPrice: '20',
        nonce: txCount,
        data: this.userContract.removeCreatedCourse.getData(address, courseId)
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...YOUR_PRIVATE_KEY...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async getBalance(address) {
    try {
      const balance = await this.userContract.getBalance(address);
      return balance.toNumber();
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async addFunds(address, amount) {
    try {
      const txCount = await this.web3.eth.getTransactionCount();
      const tx = {
        from: '0x...YOUR_ACCOUNT_ADDRESS...',
        to: this.userContract.address,
        value: '0',
        gas: '200000',
        gasPrice: '20',
        nonce: txCount,
        data: this.userContract.addFunds.getData(address, amount)
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...YOUR_PRIVATE_KEY...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async subtractFunds(address, amount) {
    try {
      const txCount = await this.web3.eth.getTransactionCount();
      const tx = {
        from: '0x...YOUR_ACCOUNT_ADDRESS...',
        to: this.userContract.address,
        value: '0',
        gas: '200000',
        gasPrice: '20',
        nonce: txCount,
        data: this.userContract.subtractFunds.getData(address, amount)
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...YOUR_PRIVATE_KEY...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

    async transferFunds(from, to, amount) {
    try {
      const txCount = await this.web3.eth.getTransactionCount();
      const tx = {
        from: '0x...YOUR_ACCOUNT_ADDRESS...',
        to: this.userContract.address,
        value: '0',
        gas: '200000',
        gasPrice: '20',
        nonce: txCount,
        data: this.userContract.transferFunds.getData(from, to, amount)
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...YOUR_PRIVATE_KEY...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt;
    } catch (error) {
      console.error(error);
      return null;
    }
  }
}

module.exports = UserController;
