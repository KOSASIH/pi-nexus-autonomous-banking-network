// mentorshipProgram.js

class MentorshipProgram {
    constructor() {
        this.mentors = []; // Store mentor profiles
        this.mentees = []; // Store mentee profiles
    }

    // Add a new mentor
    addMentor(name, expertise, bio) {
        const mentor = {
            id: this.mentors.length + 1,
            name,
            expertise,
            bio,
        };
        this.mentors.push(mentor);
        console.log(`Mentor added: ${name}`);
    }

    // Add a new mentee
    addMentee(name, goals) {
        const mentee = {
            id: this.mentees.length + 1,
            name,
            goals,
        };
        this.mentees.push(mentee);
        console.log(`Mentee added: ${name}`);
    }

    // Match mentee with a mentor based on expertise
    matchMenteeToMentor(menteeId) {
        const mentee = this.mentees.find(m => m.id === menteeId);
        if (!mentee) {
            console.log(`Mentee with ID ${menteeId} not found.`);
            return null;
        }

        const matchedMentor = this.mentors.find(mentor => 
            mentor.expertise.toLowerCase() === mentee.goals.toLowerCase()
        );

        if (matchedMentor) {
            console.log(`Mentee ${mentee.name} matched with Mentor ${matchedMentor.name}.`);
            return matchedMentor;
        } else {
            console.log(`No suitable mentor found for mentee ${mentee.name}.`);
            return null;
        }
    }

    // Get all mentors
    getAllMentors() {
        return this.mentors;
    }

    // Get all mentees
    getAllMentees() {
        return this.mentees;
    }
}

// Example usage
const mentorshipProgram = new MentorshipProgram();
mentorshipProgram.addMentor('Alice Johnson', 'Investing', 'Expert in stock market investments and portfolio management.');
mentorshipProgram.addMentor('Bob Smith', 'Budgeting', 'Specialist in personal finance and budgeting strategies.');

mentorshipProgram.addMentee('Charlie Brown', 'Investing');
mentorshipProgram.addMentee('Diana Prince', 'Budgeting');

const matchedMentor = mentorshipProgram.matchMenteeToMentor(1); // Match Charlie Brown
console.log('Matched Mentor:', matchedMentor);
