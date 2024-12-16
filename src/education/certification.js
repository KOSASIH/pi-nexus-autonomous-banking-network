// certification.js

class Certification {
    constructor() {
        this.certificationPrograms = []; // Store certification programs
    }

    // Create a new certification program
    createCertification(title, description, requirements, duration) {
        const certification = {
            id: this.certificationPrograms.length + 1,
            title,
            description,
            requirements,
            duration,
            enrolledUsers: [],
            completedUsers: [],
        };
        this.certificationPrograms.push(certification);
        console.log(`Certification program created:`, certification);
        return certification;
    }

    // Get all certification programs
    getAllCertifications() {
        return this.certificationPrograms;
    }

    // Enroll a user in a certification program
    enrollUser (certificationId, userId) {
        const certification = this.certificationPrograms.find(c => c.id === certificationId);
        if (certification) {
            if (!certification.enrolledUsers.includes(userId)) {
                certification.enrolledUsers.push(userId);
                console.log(`User  ${userId} enrolled in certification:`, certification.title);
                return certification;
            } else {
                throw new Error('User  is already enrolled in this certification program.');
            }
        } else {
            throw new Error('Certification program not found.');
        }
    }

    // Mark a user as completed in a certification program
    completeCertification(certificationId, userId) {
        const certification = this.certificationPrograms.find(c => c.id === certificationId);
        if (certification) {
            if (certification.enrolledUsers.includes(userId) && !certification.completedUsers.includes(userId)) {
                certification.completedUsers.push(userId);
                console.log(`User  ${userId} completed certification:`, certification.title);
                return certification;
            } else {
                throw new Error('User  is not enrolled or has already completed this certification.');
            }
        } else {
            throw new Error('Certification program not found.');
        }
    }

    // Get users who completed a certification program
    getCompletedUsers(certificationId) {
        constcertification = this.certificationPrograms.find(c => c.id === certificationId);
        if (certification) {
            return certification.completedUsers;
        } else {
            throw new Error('Certification program not found.');
        }
    }
}

// Example usage
const certificationManager = new Certification();
const newCertification = certificationManager.createCertification(
    'Financial Analyst Certification',
    'A comprehensive program to become a certified financial analyst.',
    'Basic understanding of finance and accounting.',
    '3 months'
);
certificationManager.createCertification(
    'Investment Management Certification',
    'Learn the principles of investment management and portfolio analysis.',
    'Completion of Financial Analyst Certification.',
    '4 months'
);

const allCertifications = certificationManager.getAllCertifications();
console.log('All Certification Programs:', allCertifications);

certificationManager.enrollUser(newCertification.id, 'user123');
certificationManager.completeCertification(newCertification.id, 'user123');
const completedUsers = certificationManager.getCompletedUsers(newCertification.id);
console.log('Users who completed certification:', completedUsers);

export default Certification;
