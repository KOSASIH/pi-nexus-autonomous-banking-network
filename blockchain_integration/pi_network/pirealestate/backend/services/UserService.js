const User = require("../models/User");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");

class UserService {
  async createUser(data) {
    const user = new User(data);
    try {
      await user.save();
      return user;
    } catch (error) {
      throw error;
    }
  }

  async getUser(id) {
    try {
      const user = await User.findById(id);
      if (!user) {
        throw new Error("User not found");
      }
      return user;
    } catch (error) {
      throw error;
    }
  }

  async updateUser(id, data) {
    try {
      const user = await User.findByIdAndUpdate(id, data, { new: true });
      if (!user) {
        throw new Error("User not found");
      }
      return user;
    } catch (error) {
      throw error;
    }
  }

  async deleteUser(id) {
    try {
      await User.findByIdAndRemove(id);
    } catch (error) {
      throw error;
    }
  }

  async authenticate(email, password) {
    try {
      const user = await User.findOne({ email });
      if (!user) {
        throw new Error("Invalid email or password");
      }
      const isValid = await bcrypt.compare(password, user.password);
      if (!isValid) {
        throw new Error("Invalid email or password");
      }
      const token = user.generateAuthToken();
      return token;
    } catch (error) {
      throw error;
    }
  }

  async authorize(token) {
    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      const user = await User.findById(decoded._id);
      if (!user) {
        throw new Error("Invalid token");
      }
      return user;
    } catch (error) {
      throw error;
    }
  }
}

module.exports = new UserService();
