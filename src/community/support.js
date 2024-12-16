// community/support.js
class SupportTicket {
    constructor(userId, issue) {
        this.id = SupportTicket.incrementId();
        this.userId = userId;
        this.issue = issue;
        this.status = 'Open'; // Status can be 'Open', 'In Progress', 'Resolved', 'Closed'
        this.createdAt = new Date();
    }

    static incrementId() {
        if (!this.currentId) {
            this.currentId = 1;
        } else {
            this.currentId++;
        }
        return this.currentId;
    }
}

class Support {
    constructor() {
        this.tickets = []; // Store support tickets
    }

    createTicket(userId, issue) {
        const ticket = new SupportTicket(userId, issue);
        this.tickets.push(ticket);
        console.log(`Support ticket created: ${ticket.id}`);
        return ticket;
    }

    getTickets() {
        return this.tickets;
    }

    getTicketById(ticketId) {
        const ticket = this.tickets.find(t => t.id === ticketId);
        if (!ticket) {
            throw new Error('Ticket not found.');
        }
        return ticket;
    }

    updateTicketStatus(ticketId, newStatus) {
        const ticket = this.getTicketById(ticketId);
        ticket.status = newStatus;
        console.log(`Ticket ${ticketId} status updated to: ${newStatus}`);
        return ticket;
    }
}

module.exports = Support;
