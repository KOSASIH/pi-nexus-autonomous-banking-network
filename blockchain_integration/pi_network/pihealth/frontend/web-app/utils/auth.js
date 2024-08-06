import { Blockchain } from './blockchain';

const blockchain = new Blockchain();

export const authenticate = async (address) => {
  try {
    const user = await blockchain.getUser(address);
    return user;
  } catch (error) {
    throw new Error('Authentication failed');
  }
};

export const authorize = async (address, role) => {
  try {
    const user = await blockchain.getUser(address);
    if (user.role === role) {
      return true;
    } else {
      throw new Error('Unauthorized');
    }
  } catch (error) {
    throw new Error('Authorization failed');
  }
};
