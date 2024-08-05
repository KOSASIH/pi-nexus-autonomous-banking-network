const CourseService = require('../server/services/CourseService');
const Course = require('../server/models/Course');

describe('CourseService', () => {
  it('should create a new course', async () => {
    const courseData = { title: 'Test Course', description: 'Test course description', price: 10 };
    const newCourse = await CourseService.createCourse(courseData);
    expect(newCourse).toBeInstanceOf(Course);
  });

  it('should get all courses', async () => {
    const courses = await CourseService.getCourses();
    expect(courses).toBeInstanceOf(Array);
  });

  it('should get a course by id', async () => {
    const courseId = '1234567890';
    const course = await CourseService.getCourseById(courseId);
    expect(course).toBeInstanceOf(Course);
  });

  it('should update a course', async () => {
    const courseId = '1234567890';
    const courseData = { title: 'Updated Course', description: 'Updated course description', price: 20 };
    const updatedCourse = await CourseService.updateCourse(courseId, courseData);
    expect(updatedCourse).toBeInstanceOf(Course);
  });

  it('should delete a course', async () => {
    const courseId = '1234567890';
    await CourseService.deleteCourse(courseId);
    expect(await Course.findById(courseId)).toBeNull();
  });
});
