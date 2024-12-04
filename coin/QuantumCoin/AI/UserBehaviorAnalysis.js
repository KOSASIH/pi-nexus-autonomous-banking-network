// UserBehaviorAnalysis.js
const fs = require('fs');
const _ = require('lodash');
const Chart = require('chart.js');
const { createCanvas } = require('canvas');

class UserBehaviorAnalysis {
    constructor(dataFile) {
        this.dataFile = dataFile;
        this.userData = [];
    }

    loadData() {
        // Load user behavior data from a JSON file
        const rawData = fs.readFileSync(this.dataFile);
        this.userData = JSON.parse(rawData);
        console.log("User data loaded successfully.");
    }

    analyzeBehavior() {
        // Analyze user behavior to find insights
        const userEngagement = {};
        
        this.userData.forEach(user => {
            const { userId, actions } = user;
            if (!userEngagement[userId]) {
                userEngagement[userId] = { totalActions: 0, uniqueActions: new Set() };
            }
            userEngagement[userId].totalActions += actions.length;
            actions.forEach(action => userEngagement[userId].uniqueActions.add(action));
        });

        // Generate insights
        const insights = Object.keys(userEngagement).map(userId => ({
            userId,
            totalActions: userEngagement[userId].totalActions,
            uniqueActions: userEngagement[userId].uniqueActions.size,
        }));

        console.log("User behavior analysis completed.");
        return insights;
    }

    visualizeData(insights) {
        // Create a bar chart to visualize user engagement
        const canvas = createCanvas(800, 400);
        const ctx = canvas.getContext('2d');

        const labels = insights.map(insight => insight.userId);
        const totalActions = insights.map(insight => insight.totalActions);

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Total Actions per User',
                    data: totalActions,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        const buffer = canvas.toBuffer('image/png');
        fs.writeFileSync('user_engagement_chart.png', buffer);
        console.log('User engagement chart saved as user_engagement_chart.png');
    }

    saveInsights(insights, outputFile) {
        fs.writeFileSync(outputFile, JSON.stringify(insights, null, 2));
        console.log(`Insights saved to ${outputFile}`);
    }
}

// Example usage
const analysis = new UserBehaviorAnalysis("user_behavior_data.json");
analysis.loadData();
const insights = analysis.analyzeBehavior();
analysis.visualizeData(insights);
analysis.saveInsights(insights, "user_behavior_insights.json");
