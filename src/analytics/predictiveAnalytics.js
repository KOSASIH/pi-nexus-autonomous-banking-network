// predictiveAnalytics.js

class PredictiveAnalytics {
    constructor() {
        this.financialData = []; // Store historical financial data
    }

    // Log financial data
    logFinancialData(dataPoint) {
        this.financialData.push(dataPoint);
        console.log(`Financial data logged:`, dataPoint);
    }

    // Predict future values based on historical data
    predictFutureValue(monthsAhead) {
        if (this.financialData.length === 0) {
            throw new Error('No financial data available for prediction.');
        }

        const lastDataPoint = this.financialData[this.financialData.length - 1];
        const averageGrowthRate = this.calculateAverageGrowthRate();

        const predictedValue = lastDataPoint.value * Math.pow(1 + averageGrowthRate, monthsAhead);
        return {
            predictedValue: predictedValue.toFixed(2),
            lastDataPoint,
            monthsAhead,
        };
    }

    // Calculate average growth rate based on historical data
    calculateAverageGrowthRate() {
        if (this.financialData.length < 2) {
            return 0; // Not enough data to calculate growth rate
        }

        let totalGrowth = 0;
        for (let i = 1; i < this.financialData.length; i++) {
            const growth = (this.financialData[i].value - this.financialData[i - 1].value) / this.financialData[i - 1].value;
            totalGrowth += growth;
        }

        return totalGrowth / (this.financialData.length - 1);
    }
}

// Example usage
const predictiveAnalytics = new PredictiveAnalytics();
predictiveAnalytics.logFinancialData({ month: '2023-01', value: 1000 });
predictiveAnalytics.logFinancialData({ month: '2023-02', value: 1100 });
predictiveAnalytics.logFinancialData({ month: '2023-03', value: 1200 });

const prediction = predictiveAnalytics.predictFutureValue(3); // Predicting 3 months ahead
console.log('Predicted Future Value:', prediction);

export default PredictiveAnalytics;
