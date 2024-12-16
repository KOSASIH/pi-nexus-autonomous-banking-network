// interactiveCourses.js

class InteractiveCourses {
    constructor() {
        this.courses = {}; // Object to store courses
        this.users = {}; // Object to store user data
    }

    // Add a new course
    addCourse(courseId, courseDetails) {
        this.courses[courseId] = {
            ...courseDetails,
            enrolledUsers: [],
            quizzes: [],
        };
        console.log(`Course "${courseDetails.title}" added with ID: ${courseId}`);
    }

    // Enroll a user in a course
    enrollUser (userId, courseId) {
        if (!this.users[userId]) {
            this.users[userId] = {
                completedCourses: [],
                quizResults: {},
            };
        }

        if (this.courses[courseId]) {
            this.courses[courseId].enrolledUsers.push(userId);
            console.log(`User  ${userId} enrolled in course "${this.courses[courseId].title}".`);
        } else {
            console.log(`Course with ID ${courseId} does not exist.`);
        }
    }

    // Add a quiz to a course
    addQuiz(courseId, quiz) {
        if (this.courses[courseId]) {
            this.courses[courseId].quizzes.push(quiz);
            console.log(`Quiz "${quiz.title}" added to course "${this.courses[courseId].title}".`);
        } else {
            console.log(`Course with ID ${courseId} does not exist.`);
        }
    }

    // Take a quiz
    takeQuiz(userId, courseId, quizTitle, answers) {
        const course = this.courses[courseId];
        if (course) {
            const quiz = course.quizzes.find(q => q.title === quizTitle);
            if (quiz) {
                const correctAnswers = quiz.questions.filter(q => q.correctAnswer === answers[q.id]);
                const score = (correctAnswers.length / quiz.questions.length) * 100;

                // Store the result
                this.users[userId].quizResults[quizTitle] = {
                    score,
                    passed: score >= quiz.passScore,
                };

                console.log(`User  ${userId} took quiz "${quizTitle}" and scored ${score}%.`);

                // Check if the user has completed the course
                if (this.checkCourseCompletion(userId, courseId)) {
                    this.issueCertificate(userId, courseId);
                }
            } else {
                console.log(`Quiz "${quizTitle}" does not exist in course "${course.title}".`);
            }
        } else {
            console.log(`Course with ID ${courseId} does not exist.`);
        }
    }

    // Check if the user has completed the course
    checkCourseCompletion(userId, courseId) {
        const course = this.courses[courseId];
        if (course) {
            const quizzesTaken = Object.keys(this.users[userId].quizResults);
            return quizzesTaken.length === course.quizzes.length;
        }
        return false;
    }

    // Issue a certificate to the user
    issueCertificate(userId, courseId) {
        const course = this.courses[courseId];
        if (course) {
            this.users[userId].completedCourses.push(courseId);
            console.log(`Certificate issued to User ${userId} for completing course "${course.title}".`);
        }
    }

    // Example usage
    exampleUsage() {
        // Adding courses
        this.addCourse('course1', { title: 'JavaScript Basics', description: 'Learn the fundamentals of JavaScript.' });
        this.addCourse('course2', { title: 'Advanced JavaScript', description: 'Deep dive into JavaScript concepts.' });

        // Enrolling users
        this.enrollUser ('user1', 'course1');
        this.enrollUser ('user2', 'course1');

        // Adding quizzes
        this.addQuiz('course1', {
            title: 'JavaScript Basics Quiz',
            questions: [
                { id: 1, question: 'What is the output of 1 + "1"?', correctAnswer: '11' },
                { id: 2, question: 'What does "NaN" stand for?', correctAnswer: 'Not a Number' },
            ],
            passScore: 50,
        });

        // Taking quizzes
        this.takeQuiz('user1', 'course1', 'JavaScript Basics Quiz', { 1: '11', 2: 'Not a Number' });
        this.takeQuiz('user2', 'course1', 'JavaScript Basics Quiz', { 1: '10 ', 2: 'Not a Number' }); // User 2 should not pass

        // Checking completion
        console.log(this.users['user1']); // Should show completed courses and quiz results
        console.log(this.users['user2']); // Should show quiz results and no completed courses
    }
}

// Example usage
const interactiveCourses = new InteractiveCourses();
interactiveCourses.exampleUsage();

export default InteractiveCourses;
