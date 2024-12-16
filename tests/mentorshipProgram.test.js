// mentorshipProgram.test.js

const MentorshipProgram = require('./mentorshipProgram'); // Assuming you have a MentorshipProgram module

describe('Mentorship Program Functionalities', () => {
    let mentorshipProgram;

    beforeEach(() => {
        mentorshipProgram = new MentorshipProgram();
    });

    test('should add a mentor successfully', () => {
        mentorshipProgram.addMentor('Alice Johnson', 'Investing', 'Expertin financial markets');
        const mentors = mentorshipProgram.getMentors();
        expect(mentors).toContainEqual(expect.objectContaining({
            name: 'Alice Johnson',
            expertise: 'Investing',
        }));
    });

    test('should add a mentee successfully', () => {
        mentorshipProgram.addMentee('Bob Smith', 'Finance');
        const mentees = mentorshipProgram.getMentees();
        expect(mentees).toContainEqual(expect.objectContaining({
            name: 'Bob Smith',
            interest: 'Finance',
        }));
    });

    test('should match mentor and mentee based on expertise', () => {
        mentorshipProgram.addMentor('Alice Johnson', 'Investing', 'Expert in financial markets');
        mentorshipProgram.addMentee('Bob Smith', 'Finance');
        const match = mentorshipProgram.matchMentorToMentee('Bob Smith');
        expect(match).toEqual(expect.objectContaining({
            mentor: 'Alice Johnson',
            mentee: 'Bob Smith',
        }));
    });

    test('should not match if no suitable mentor is found', () => {
        mentorshipProgram.addMentee('Charlie Brown', 'Art');
        const match = mentorshipProgram.matchMentorToMentee('Charlie Brown');
        expect(match).toBeNull(); // Assuming no mentor available for Art
    });
});
