// sentimentAnalysis.js

class SentimentAnalysis {
    constructor() {
        this.positiveWords = ['good', 'great', 'excellent', 'happy', 'love', 'fantastic', 'amazing'];
        this.negativeWords = ['bad', 'terrible', 'hate', 'sad', 'awful', 'poor', 'disappointing'];
    }

    // Analyze sentiment of a given text
    analyzeSentiment(text) {
        const words = text.toLowerCase().split(/\s+/);
        let positiveCount = 0;
        let negativeCount = 0;

        words.forEach(word => {
            if (this.positiveWords.includes(word)) {
                positiveCount++;
            } else if (this.negativeWords.includes(word)) {
                negativeCount++;
            }
        });

        const totalWords = positiveCount + negativeCount;
        let sentimentScore = 0;

        if (totalWords > 0) {
            sentimentScore = (positiveCount - negativeCount) / totalWords;
        }

        return {
            sentimentScore,
            sentiment: sentimentScore > 0 ? 'Positive' : sentimentScore < 0 ? 'Negative' : 'Neutral',
            positiveCount,
            negativeCount,
        };
    }

    // Analyze multiple feedbacks
    analyzeMultipleFeedbacks(feedbacks) {
        return feedbacks.map(feedback => ({
            feedback,
            analysis: this.analyzeSentiment(feedback),
        }));
    }
}

// Example usage
const sentimentAnalyzer = new SentimentAnalysis();
const feedbacks = [
    "I love this product! It's fantastic.",
    "This is the worst experience I've ever had.",
    "The service was okay, nothing special.",
];

const results = sentimentAnalyzer.analyzeMultipleFeedbacks(feedbacks);
console.log('Sentiment Analysis Results:', results);
