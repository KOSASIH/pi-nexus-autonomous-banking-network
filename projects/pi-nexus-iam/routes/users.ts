import express, { Request, Response, NextFunction } from 'express';
import { User } from '../models/user';
import { Role } from '../enums/role.enum';
import { AccessControl } from '../models/access_control';

const router = express.Router();

router.get('/', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const users = await User.find().populate('roles');
    res.json(users);
  } catch (error) {
    next(error);
  }
});

router.get('/:id', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const id = req.params.id;
    const user = await User.findById(id).populate('roles');
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    res.json(user);
  } catch (error) {
    next(error);
  }
});

router.put('/:id', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const id = req.params.id;
    const { username, email, roles } = req.body;
    const user = await User.findByIdAndUpdate(id, { username, email, roles }, { new: true });
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    res.json(user);
  } catch (error) {
    next(error);
  }
});

router.delete('/:id', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const id = req.params.id;
    await User.findByIdAndRemove(id);
    res.json({ message: 'User deleted successfully' });
  } catch (error) {
    next(error);
  }
});

router.post('/:id/roles', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const id = req.params.id;
    const { roles } = req.body;
    const user = await User.findById(id);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    const accessControl = await AccessControl.findOne({ role: roles });
        if (!accessControl) {
      return res.status(404).json({ error: 'Access control not found' });
    }
    user.roles.push(accessControl.role);
    await user.save();
    res.json({ message: 'Roles updated successfully' });
  } catch (error) {
    next(error);
  }
});

router.delete('/:id/roles/:roleId', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const id = req.params.id;
    const roleId = req.params.roleId;
    const user = await User.findById(id);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    const index = user.roles.indexOf(roleId);
    if (index === -1) {
      return res.status(404).json({ error: 'Role not found' });
    }
    user.roles.splice(index, 1);
    await user.save();
    res.json({ message: 'Role removed successfully' });
  } catch (error) {
    next(error);
  }
});

export default router;
