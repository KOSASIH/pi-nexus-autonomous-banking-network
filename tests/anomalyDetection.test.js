// anomalyDetection.test.js

import AnomalyDetection from './anomalyDetection'; // Assuming you have an AnomalyDetection class/module

describe('Anomaly Detection', () => {
    let anomalyDetector;

    beforeEach(() => {
        anomalyDetector = new AnomalyDetection();
    });

    test('should detect an anomaly in the data', () => {
        const data = [1, 2, 3, 100, 5]; // 100 is an anomaly
        const result = anomalyDetector.detect(data);
        expect(result).toEqual([100]);
    });

    test('should return an empty array if no anomalies are found', () => {
        const data = [1, 2, 3, 4, 5];
        const result = anomalyDetector.detect(data);
        expect(result).toEqual([]);
    });

    test('should throw error if data is not provided', () => {
        expect(() => anomalyDetector.detect()).toThrow('Data is required');
    });
});
