const UserService = require('../server/services/UserService');
const User = require('../server/models/User');
const bcrypt = require('bcrypt');

describe('UserService', () => {
  beforeEach(async () => {
    await User.deleteMany({}); // clear the users collection before each test
  });

  it('should create a new user', async () => {
    const userData = {
      email: 'test@example.com',
      password: 'password123',
      name: 'John Doe'
    };
    const newUser = await UserService.createUser(userData);
    expect(newUser).toBeInstanceOf(User);
    expect(newUser.email).toBe(userData.email);
    expect(newUser.name).toBe(userData.name);
    expect(await bcrypt.compare(userData.password, newUser.password)).toBe(true);
  });

  it('should get a user by email', async () => {
    const userData = {
      email: 'test@example.com',
      password: 'password123',
      name: 'John Doe'
    };
    const user = await UserService.createUser(userData);
    const foundUser = await UserService.getUserByEmail(userData.email);
    expect(foundUser).toBeInstanceOf(User);
    expect(foundUser.email).toBe(userData.email);
    expect(foundUser.name).toBe(userData.name);
  });

  it('should get a user by id', async () => {
    const userData = {
      email: 'test@example.com',
      password: 'password123',
      name: 'John Doe'
    };
    const user = await UserService.createUser(userData);
    const foundUser = await UserService.getUserById(user.id);
    expect(foundUser).toBeInstanceOf(User);
    expect(foundUser.email).toBe(userData.email);
    expect(foundUser.name).toBe(userData.name);
  });

  it('should update a user', async () => {
    const userData = {
      email: 'test@example.com',
      password: 'password123',
      name: 'John Doe'
    };
    const user = await UserService.createUser(userData);
    const updatedUserData = {
      email: 'updated@example.com',
      name: 'Jane Doe'
    };
    const updatedUser = await UserService.updateUser(user.id, updatedUserData);
    expect(updatedUser).toBeInstanceOf(User);
    expect(updatedUser.email).toBe(updatedUserData.email);
    expect(updatedUser.name).toBe(updatedUserData.name);
  });

  it('should delete a user', async () => {
    const userData = {
      email: 'test@example.com',
      password: 'password123',
      name: 'John Doe'
    };
    const user = await UserService.createUser(userData);
    await UserService.deleteUser(user.id);
    expect(await User.findById(user.id)).toBeNull();
  });
});
