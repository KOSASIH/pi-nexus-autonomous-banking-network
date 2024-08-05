const express = require('express');
const router = express.Router();
const CourseService = require('../services/CourseService');

const courseService = new CourseService();

router.post('/create', async (req, res) => {
  try {
    const { title, description, price } = req.body;
    const courseId = await courseService.createCourse(title, description, price);
    res.json({ courseId });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create course' });
  }
});

router.get('/:id', async (req, res) => {
  try {
    const courseId = req.params.id;
    const courseData = await courseService.getCourse(courseId);
    res.json(courseData);
  } catch (error) {
    console.error(error);
    res.status(404).json({ error: 'Course not found' });
  }
});

router.put('/:id', async (req, res) => {
  try {
    const courseId = req.params.id;
    const { title, description, price } = req.body;
    await courseService.updateCourse(courseId, title, description, price);
    res.json({ message: 'Course updated successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to update course' });
  }
});

router.delete('/:id', async (req, res) => {
  try {
    const courseId = req.params.id;
    await courseService.deleteCourse(courseId);
    res.json({ message: 'Course deleted successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to delete course' });
  }
});

router.post('/:id/enroll', async (req, res) => {
  try {
    const courseId = req.params.id;
    const userId = req.body.userId;
    await courseService.enrollUserInCourse(userId, courseId);
    res.json({ message: 'User enrolled in course successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to enroll user in course' });
  }
});

router.post('/:id/unenroll', async (req, res) => {
  try {
    const courseId = req.params.id;
    const userId = req.body.userId;
    await courseService.unenrollUserFromCourse(userId, courseId);
    res.json({ message: 'User unenrolled from course successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to unenroll user from course' });
  }
});

module.exports = router;
