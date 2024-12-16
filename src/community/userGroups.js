// userGroups.js

class UserGroups {
    constructor() {
        this.groups = {}; // Store groups by groupId
        this.userGroups = {}; // Store user memberships
    }

    // Create a new group
    createGroup(groupId, groupName, ownerId) {
        if (!this.groups[groupId]) {
            this.groups[groupId] = {
                groupName,
                ownerId,
                members: [],
            };
            console.log(`Group "${groupName}" created with ID: ${groupId}`);
        } else {
            console.log(`Group ID ${groupId} already exists.`);
        }
    }

    // Join a group
    joinGroup(userId, groupId) {
        if (this.groups[groupId]) {
            if (!this.userGroups[userId]) {
                this.userGroups[userId] = [];
            }
            if (!this.userGroups[userId].includes(groupId)) {
                this.userGroups[userId].push(groupId);
                this.groups[groupId].members.push(userId);
                console.log(`User  ${userId} joined group "${this.groups[groupId].groupName}".`);
            } else {
                console.log(`User  ${userId} is already a member of group "${this.groups[groupId].groupName}".`);
            }
        } else {
            console.log(`Group ID ${groupId} does not exist.`);
        }
    }

    // Leave a group
    leaveGroup(userId, groupId) {
        if (this.groups[groupId] && this.userGroups[userId]) {
            const groupIndex = this.userGroups[userId].indexOf(groupId);
            const memberIndex = this.groups[groupId].members.indexOf(userId);

            if (groupIndex !== -1) {
                this.userGroups[userId].splice(groupIndex, 1);
                this.groups[groupId].members.splice(memberIndex, 1);
                console.log(`User  ${userId} left group "${this.groups[groupId].groupName}".`);
            } else {
                console.log(`User  ${userId} is not a member of group "${this.groups[groupId].groupName}".`);
            }
        } else {
            console.log(`Group ID ${groupId} or User ID ${userId} does not exist.`);
        }
    }

    // Get group details
    getGroupDetails(groupId) {
        return this.groups[groupId] || null;
    }

    // Get user groups
    getUser Groups(userId) {
        return this.userGroups[userId] || [];
    }
}

// Example usage
const userGroups = new UserGroups();
userGroups.createGroup('group1', 'Tech Enthusiasts', 'user1');
userGroups.createGroup('group2', 'Book Lovers', 'user2');

userGroups.joinGroup('user3', 'group1');
userGroups.joinGroup('user1', 'group2');
userGroups.leaveGroup('user3', 'group1');

console.log('Group Details:', userGroups.getGroupDetails('group1'));
console.log('User  Groups for user1:', userGroups.getUser Groups('user1'));
