// communityModeration.js

class CommunityModeration {
    constructor() {
        this.reports = []; // Store reports
        this.bannedUsers = new Set(); // Store banned users
    }

    // Report a user
    reportUser (reporterId, reportedId, reason) {
        this.reports.push({
            reporterId,
            reportedId,
            reason,
            timestamp: new Date(),
        });
        console.log(`User  ${reporterId} reported user ${reportedId} for: ${reason}`);
    }

    // Review reports
    reviewReports() {
        return this.reports;
    }

    // Ban a user
    banUser (userId) {
        this.bannedUsers.add(userId);
        console.log(`User  ${userId} has been banned from the community.`);
    }

    // Unban a user
    unbanUser (userId) {
        this.bannedUsers.delete(userId);
        console.log(`User  ${userId} has been unbanned from the community.`);
    }

    // Check if a user is banned
    isUser Banned(userId) {
        return this.bannedUsers.has(userId);
    }
}

// Example usage
const moderation = new CommunityModeration();
moderation.reportUser ('user1', 'user2', 'Inappropriate behavior');
moderation.reportUser ('user3', ' user4', 'Spam content');

moderation.banUser ('user2');
console.log('Current Reports:', moderation.reviewReports());
console.log('Is user2 banned?', moderation.isUser Banned('user2'));

moderation.unbanUser ('user2');
console.log('Is user2 banned after unbanning?', moderation.isUser Banned('user2'));
