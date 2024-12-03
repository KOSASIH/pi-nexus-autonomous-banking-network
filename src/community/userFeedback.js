// userFeedback.js

class UserFeedback {
    constructor() {
        this.feedbackList = []; // Store user feedback
    }

    // Collect user feedback
    collectFeedback(userId, feedbackText) {
        const feedbackEntry = {
            userId,
            feedbackText,
            timestamp: new Date(),
        };
        this.feedbackList.push(feedbackEntry);
        console.log(`Feedback collected from user ${userId}:`, feedbackEntry);
        return feedbackEntry;
    }

    // Get all feedback
    getAllFeedback() {
        return this.feedbackList;
    }

    // Get feedback by user
    getFeedbackByUser (userId) {
        return this.feedbackList.filter(feedback => feedback.userId === userId);
    }

    // Analyze feedback for common themes
    analyzeFeedback() {
        const feedbackThemes = {};
        this.feedbackList.forEach(entry => {
            const words = entry.feedbackText.split(' ');
            words.forEach(word => {
                const cleanedWord = word.toLowerCase().replace(/[.,!?]/g, '');
                feedbackThemes[cleanedWord] = (feedbackThemes[cleanedWord] || 0) + 1;
            });
        });

        return feedbackThemes;
    }
}

// Example usage
const userFeedbackManager = new UserFeedback();
userFeedbackManager.collectFeedback('user123', 'Great app, but I would love more features!');
userFeedbackManager.collectFeedback('user456', 'The interface is confusing.');
userFeedbackManager.collectFeedback('user123', 'I appreciate the updates!');

const allFeedback = userFeedbackManager.getAllFeedback();
console.log('All User Feedback:', allFeedback);

const user123Feedback = userFeedbackManager.getFeedbackByUser ('user123');
console.log('Feedback from user123:', user123Feedback);

const feedbackAnalysis = userFeedbackManager.analyzeFeedback();
console.log('Feedback Analysis:', feedbackAnalysis);

export default UserFeedback;
