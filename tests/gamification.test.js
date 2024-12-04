// gamification.test.js

const Gamification = require('./gamification'); // Assuming you have a Gamification module

describe('Gamification Features', () => {
    let gamification;

    beforeEach(() => {
        gamification = new Gamification();
    });

    test('should award points for completing a task', () => {
        gamification.completeTask('user1', 'task1');
        const points = gamification.getUser Points('user1');
        expect(points).toBeGreaterThan(0);
    });

    test('should unlock a badge after earning enough points', () => {
        gamification.completeTask('user1', 'task1');
        gamification.completeTask('user1', 'task2');
        const badges = gamification.getUser Badges('user1');
        expect(badges).toContain('Task Master');
    });

    test('should not award points for duplicate task completion', () => {
        gamification.completeTask('user1', 'task1');
        gamification.completeTask('user1', 'task1'); // Duplicate
        const points = gamification.getUser Points('user1');
        expect(points).toBe(10); // Assuming task1 is worth 10 points
    });
});
