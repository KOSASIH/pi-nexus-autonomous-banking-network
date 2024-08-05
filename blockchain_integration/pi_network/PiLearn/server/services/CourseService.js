const Course = require('../models/Course');
const CourseContract = require('../smart-contracts/CourseContract');

class CourseService {
  async getCourses() {
    const courses = await Course.find().exec();
    return courses;
  }

  async getCourseById(id) {
    const course = await Course.findById(id).exec();
    return course;
  }

  async createCourse(courseData) {
    const newCourse = new Course(courseData);
    await newCourse.save();
    await CourseContract.createCourse(courseData.title, courseData.description, courseData.price);
    return newCourse;
  }

  async updateCourse(id, courseData) {
    const course = await Course.findByIdAndUpdate(id, courseData, { new: true }).exec();
    return course;
  }

  async deleteCourse(id) {
    await Course.findByIdAndRemove(id).exec();
    await CourseContract.deleteCourse(id);
  }
}

module.exports = CourseService;
