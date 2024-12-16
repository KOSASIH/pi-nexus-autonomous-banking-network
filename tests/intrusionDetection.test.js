// intrusionDetection.test.js

import IntrusionDetection from './intrusionDetection'; // Import the module to be tested

describe('IntrusionDetection', () => {
    let intrusionDetection;

    beforeEach(() => {
        intrusionDetection = new IntrusionDetection();
    });

    test('should detect an intrusion attempt', () => {
        const result = intrusionDetection.detectIntrusion({ type: 'brute_force', source: '192.168.1.1' });
        expect(result).toBe(true);
    });

    test('should not detect normal traffic as intrusion', () => {
        const result = intrusionDetection.detectIntrusion({ type: 'normal', source: '192.168.1.1' });
        expect(result).toBe(false);
    });

    test('should log intrusion attempts', () => {
        intrusionDetection.detectIntrusion({ type: 'brute_force', source: '192.168.1.1' });
        const logs = intrusionDetection.getLogs();
        expect(logs).toHaveLength(1);
        expect(logs[0]).toMatchObject({ type: 'brute_force', source: '192.168.1.1' });
    });
});
