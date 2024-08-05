const express = require('express');
const router = express.Router();
const CourseController = require('../controllers/CourseController');

router.get('/courses', CourseController.getCourses);
router.get('/courses/:id', CourseController.getCourseById);
router.post('/courses', CourseController.createCourse);
router.put('/courses/:id', CourseController.updateCourse);
router.delete('/courses/:id', CourseController.deleteCourse);

module.exports = router;
