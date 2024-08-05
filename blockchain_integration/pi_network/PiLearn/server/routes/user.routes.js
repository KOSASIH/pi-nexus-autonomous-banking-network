const express = require('express');
const router = express.Router();
const UserService = require('../services/UserService');

const userService = new UserService();

router.post('/create', async (req, res) => {
  try {
    const { username, email, password } = req.body;
    const userAddress = await userService.createUser(username, email, password);
    res.json({ userAddress });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create user' });
  }
});

router.get('/:address', async (req, res) => {
  try {
    const userAddress = req.params.address;
    const userData = await userService.getUser(userAddress);
    res.json(userData);
  } catch (error) {
    console.error(error);
    res.status(404).json({ error: 'User not found' });
  }
});

router.put('/:address', async (req, res) => {
  try {
    const userAddress = req.params.address;
    const { username, email, password } = req.body;
    await userService.updateUser(userAddress, username, email, password);
    res.json({ message: 'User updated successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to update user' });
  }
});

router.post('/:address/add-funds', async (req, res) => {
  try {
    const userAddress = req.params.address;
    const amount = req.body.amount;
    await userService.addFunds(userAddress, amount);
    res.json({ message: 'Funds added successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to add funds' });
  }
});

router.post('/:address/transfer-funds', async (req, res) => {
  try {
    const fromAddress = req.params.address;
    const toAddress = req.body.toAddress;
    const amount = req.body.amount;
    await userService.transferFunds(fromAddress, toAddress, amount);
    res.json({ message: 'Funds transferred successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to transfer funds' });
  }
});

router.get('/:address/courses', async (req, res) => {
  try {
    const userAddress = req.params.address;
    const courses = await userService.getUserCourses(userAddress);
    res.json(courses);
  } catch (error) {
    console.error(error);
    res.status(404).json({ error: 'User not found' });
  }
});

router.post('/:address/enroll', async (req, res) => {
  try {
    const userAddress = req.params.address;
    const courseId = req.body.courseId;
    await userService.enrollUserInCourse(userAddress, courseId);
    res.json({ message: 'User enrolled in course successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to enroll user in course' });
  }
});

router.post('/:address/unenroll', async (req, res) => {
  try {
    const userAddress = req.params.address;
    const courseId = req.body.courseId;
    await userService.unenrollUserFromCourse(userAddress, courseId);
    res.json({ message: 'User unenrolled from course successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to unenroll user from course' });
  }
});

module.exports = router;
