// events.js

class Events {
    constructor() {
        this.eventsList = []; // Store community events
    }

    // Create a new event
    createEvent(title, description, date, location) {
        const event = {
            id: this.eventsList.length + 1,
            title,
            description,
            date,
            location,
            attendees: [],
        };
        this.eventsList.push(event);
        console.log(`Event created:`, event);
        return event;
    }

    // Get all events
    getAllEvents() {
        return this.eventsList;
    }

    // Register a user for an event
    registerForEvent(eventId, userId) {
        const event = this.eventsList.find(e => e.id === eventId);
        if (event) {
            if (!event.attendees.includes(userId)) {
                event.attendees.push(userId);
                console.log(`User ${userId} registered for event:`, event.title);
                return event;
            } else {
                throw new Error('User is already registered for this event.');
            }
        } else {
            throw new Error('Event not found.');
        }
    }

    // Get attendees for an event
    getEventAttendees(eventId) {
        const event = this.eventsList.find(e => e.id === eventId);
        if (event) {
            return event.attendees;
        } else {
            throw new Error('Event not found.');
        }
    }
}

// Example usage
const eventsManager = new Events();
const newEvent = eventsManager.createEvent('Webinar on User Experience', 'Join us for a discussion on improving user experience.', '2023-10-15', 'Online');
eventsManager.createEvent('Community Meetup', 'A chance to meet and network with other users.', '2023-11-01', 'Local Park');

const allEvents = eventsManager.getAllEvents();
console.log('All Community Events:', allEvents);

eventsManager.registerForEvent(newEvent.id, 'user123');
const attendees = eventsManager.getEventAttendees(newEvent.id);
console.log('Attendees for event:', attendees);

export default Events;
