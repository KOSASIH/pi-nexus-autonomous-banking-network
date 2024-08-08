import express from 'express';
import { createUser, getUsers, getUser, updateUser, deleteUser } from '../services/userService';

const router = express.Router();

router.post('/', createUser);
router.get('/', getUsers);
router.get('/:id', getUser);
router.patch('/:id', updateUser);
router.delete('/:id', deleteUser);

export default router;
