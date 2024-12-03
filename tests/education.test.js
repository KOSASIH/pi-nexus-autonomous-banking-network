// tests/education.test.js
const Education = require('../education'); // Assuming you have an education module

describe('Education Module', () => {
    let education;

    beforeEach(() => {
        education = new Education();
    });

    test('should add a new course correctly', () => {
        education.addCourse('JavaScript Basics');
        expect(education.getCourses()).toContain('JavaScript Basics');
    });

    test('should remove a course correctly', () => {
        education.addCourse('JavaScript Basics');
        education.removeCourse('JavaScript Basics');
        expect(education.getCourses()).not.toContain('JavaScript Basics');
    });
});
