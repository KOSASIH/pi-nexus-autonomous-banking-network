const UserController = require('./UserController');

class UserService {
  constructor() {
    this.userController = new UserController();
  }

  async createUser(username, email, password) {
    try {
      const userAddress = await this.userController.createUser(username, email, password);
      return userAddress;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async getUser(address) {
    try {
      const userData = await this.userController.getUser(address);
      return userData;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async updateUser(address, username, email, password) {
    try {
      await this.userController.updateUser(address, username, email, password);
      return true;
    } catch (error) {
      console.error(error);
      return false;
    }
  }

  async addCreatedCourse(address, courseId) {
    try {
      await this.userController.addCreatedCourse(address, courseId);
      return true;
    } catch (error) {
      console.error(error);
      return false;
    }
  }

  async removeOwnedCourse(address, courseId) {
    try {
      await this.userController.removeOwnedCourse(address, courseId);
      return true;
    } catch (error) {
      console.error(error);
      return false;
    }
  }

  async removeEnrolledCourse(address, courseId) {
    try {
      await this.userController.removeEnrolledCourse(address, courseId);
      return true;
    } catch (error) {
      console.error(error);
      return false;
    }
  }

  async removeCreatedCourse(address, courseId) {
    try {
      await this.userController.removeCreatedCourse(address, courseId);
      return true;
    } catch (error) {
      console.error(error);
      return false;
    }
  }

  async getBalance(address) {
    try {
      const balance = await this.userController.getBalance(address);
      return balance;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  async addFunds(address, amount) {
    try {
      await this.userController.addFunds(address, amount);
      return true;
    } catch (error) {
      console.error(error);
      return false;
    }
  }

  async subtractFunds(address, amount) {
    try {
      await this.userController.subtractFunds(address, amount);
      return true;
    } catch (error) {
      console.error(error);
      return false;
    }
  }

  async transferFunds(from, to, amount) {
    try {
      await this.userController.transferFunds(from, to, amount);
      return true;
    } catch (error) {
      console.error(error);
      return false;
    }
  }
}

module.exports = UserService;
