// gamification.js

class Gamification {
    constructor() {
        this.users = {}; // Store user data
        this.achievements = {}; // Store achievements
        this.leaderboard = []; // Store leaderboard entries
    }

    // Register a new user
    registerUser (userId) {
        if (!this.users[userId]) {
            this.users[userId] = {
                points: 0,
                achievements: [],
            };
            console.log(`User  ${userId} registered.`);
        } else {
            console.log(`User  ${userId} is already registered.`);
        }
    }

    // Award points to a user
    awardPoints(userId, points) {
        if (this.users[userId]) {
            this.users[userId].points += points;
            console.log(`Awarded ${points} points to user ${userId}.`);
            this.checkAchievements(userId);
            this.updateLeaderboard(userId);
        } else {
            console.log(`User  ${userId} not found.`);
        }
    }

    // Define an achievement
    defineAchievement(achievementId, description, pointsRequired) {
        this.achievements[achievementId] = { description, pointsRequired };
        console.log(`Achievement defined: ${description} (Requires ${pointsRequired} points)`);
    }

    // Check if a user has unlocked any achievements
    checkAchievements(userId) {
        const userPoints = this.users[userId].points;
        for (const [achievementId, achievement] of Object.entries(this.achievements)) {
            if (userPoints >= achievement.pointsRequired && !this.users[userId].achievements.includes(achievementId)) {
                this.users[userId].achievements.push(achievementId);
                console.log(`User  ${userId} unlocked achievement: ${achievement.description}`);
            }
        }
    }

    // Update the leaderboard
    updateLeaderboard(userId) {
        const userPoints = this.users[userId].points;
        const entryIndex = this.leaderboard.findIndex(entry => entry.userId === userId);

        if (entryIndex !== -1) {
            this.leaderboard[entryIndex].points = userPoints; // Update existing entry
        } else {
            this.leaderboard.push({ userId, points: userPoints }); // Add new entry
        }

        // Sort leaderboard by points in descending order
        this.leaderboard.sort((a, b) => b.points - a.points);
        console.log(`Leaderboard updated for user ${userId}.`);
    }

    // Get the current leaderboard
    getLeaderboard() {
        return this.leaderboard.map(entry => ({
            userId: entry.userId,
            points: entry.points,
        }));
    }

    // Example usage
    static exampleUsage() {
        const gamification = new Gamification();
        gamification.registerUser ('user1');
        gamification.registerUser ('user2');

        gamification.defineAchievement('first_points', 'Earn your first points', 10);
        gamification.defineAchievement('ten_points', 'Earn 10 points', 10);
        gamification.defineAchievement('fifty_points', 'Earn 50 points', 50);

        gamification.awardPoints('user1', 15);
        gamification.awardPoints('user2', 5);
        gamification.awardPoints('user1', 40);

        console.log('Current Leaderboard:', gamification.getLeaderboard());
    }
}

// Run example usage
Gamification.exampleUsage();
