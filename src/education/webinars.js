// webinars.js

class Webinars {
    constructor() {
        this.webinarList = []; // Store educational webinars
    }

    // Create a new webinar
    createWebinar(title, description, date, duration, presenter) {
        const webinar = {
            id: this.webinarList.length + 1,
            title,
            description,
            date,
            duration,
            presenter,
            attendees: [],
        };
        this.webinarList.push(webinar);
        console.log(`Webinar created:`, webinar);
        return webinar;
    }

    // Get all webinars
    getAllWebinars() {
        return this.webinarList;
    }

    // Register a user for a webinar
    registerForWebinar(webinarId, userId) {
        const webinar = this.webinarList.find(w => w.id === webinarId);
        if (webinar) {
            if (!webinar.attendees.includes(userId)) {
                webinar.attendees.push(userId);
                console.log(`User  ${userId} registered for webinar:`, webinar.title);
                return webinar;
            } else {
                throw new Error('User  is already registered for this webinar.');
            }
        } else {
            throw new Error('Webinar not found.');
        }
    }

    // Get attendees for a webinar
    getWebinarAttendees(webinarId) {
        const webinar = this.webinarList.find(w => w.id === webinarId);
        if (webinar) {
            return webinar.attendees;
        } else {
            throw new Error('Webinar not found.');
        }
    }
}

// Example usage
const webinarsManager = new Webinars();
const newWebinar = webinarsManager.createWebinar(
    'Understanding Financial Literacy',
    'Join us for an informative session on financial literacy basics.',
    '2023-10-20',
    '1 hour',
    'Jane Doe'
);
webinarsManager.createWebinar(
    'Investing 101',
    'Learn the fundamentals of investing and how to get started.',
    '2023-11-05',
    '1.5 hours',
    'John Smith'
);

const allWebinars = webinarsManager.getAllWebinars();
console.log('All Educational Webinars:', allWebinars);

webinarsManager.registerForWebinar(newWebinar.id, 'user123');
const attendees = webinarsManager.getWebinarAttendees(newWebinar.id);
console.log('Attendees for webinar:', attendees);

export default Webinars;
