import { Request, Response } from 'express';
import { getUsers, getUser, createUser, updateUser, deleteUser } from '../services/user.service';
import { validate } from '../utils/validation.util';
import { UserRequest } from '../types/user.type';

const userController = {
  async getUsers(req: Request, res: Response) {
    try {
      const users = await getUsers();
      res.json(users);
    } catch (error) {
      res.status(500).json({ error: 'Failed to retrieve users' });
    }
  },

  async getUser(req: Request, res: Response) {
    try {
      const id = req.params.id;
      const user = await getUser(id);
      if (!user) {
        res.status(404).json({ error: 'User not found' });
      } else {
        res.json(user);
      }
    } catch (error) {
      res.status(500).json({ error: 'Failed to retrieve user' });
    }
  },

  async createUser(req: Request, res: Response) {
    try {
      const { username, email, password } = req.body;
      const user = await createUser(username, email, password);
      res.json({ message: 'User created successfully' });
    } catch (error) {
      res.status(400).json({ error: 'Failed to create user' });
    }
  },

  async updateUser(req: Request, res: Response) {
  try {
    const id = req.params.id;
    const { username, email, password } = req.body;
    const user = await updateUser(id, username, email, password);
    res.json({ message: 'User updated successfully' });
  } catch (error) {
    res.status(400).json({ error: 'Failed to update user' });
  }
},

async deleteUser(req: Request, res: Response) {
  try {
    const id = req.params.id;
    await deleteUser(id);
    res.json({ message: 'User deleted successfully' });
  } catch (error) {
    res.status(400).json({ error: 'Failed to delete user' });
  }
},
};

export default userController;

async function getUsers() {
  const users = await User.find().exec();
  return users;
}

async function getUser(id: string) {
  const user = await User.findById(id).exec();
  return user;
}

async function createUser(username: string, email: string, password: string) {
  const user = new User({ username, email, password });
  await user.save();
  return user;
}

async function updateUser(id: string, username: string, email: string, password: string) {
  const user = await User.findByIdAndUpdate(id, { username, email, password }, { new: true });
  return user;
}

async function deleteUser(id: string) {
  await User.findByIdAndRemove(id);
}
