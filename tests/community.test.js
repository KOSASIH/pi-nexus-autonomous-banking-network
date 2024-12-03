// tests/community.test.js
const Community = require('../community'); // Assuming you have a community module

describe('Community Module', () => {
    let community;

    beforeEach(() => {
        community = new Community();
    });

    test('should add a new post correctly', () => {
        community.addPost('Hello World');
        expect(community.getPosts()).toContain('Hello World');
    });

    test('should delete a post correctly', () => {
        community.addPost('Hello World');
        community.deletePost('Hello World');
        expect(community.getPosts()).not.toContain('Hello World');
    });
});
