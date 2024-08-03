# course_management.py
class Course:
    def __init__(self, course_id, course_name, course_description):
        self.course_id = course_id
        self.course_name = course_name
        self.course_description = course_description

class CourseManagement:
    def __init__(self):
        self.courses = []

    def add_course(self, course):
        self.courses.append(course)

    def get_courses(self):
        return self.courses
