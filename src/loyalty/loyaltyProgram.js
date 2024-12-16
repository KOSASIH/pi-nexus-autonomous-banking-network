// loyalty/loyaltyProgram.js

class LoyaltyProgram {
    constructor(name, description, pointsPerPurchase) {
        this.name = name;
        this.description = description;
        this.pointsPerPurchase = pointsPerPurchase;
        this.members = new Map(); // Store userId and their points
    }

    addMember(userId) {
        if (!this.members.has(userId)) {
            this.members.set(userId, 0); // Initialize points to 0
        }
    }

    removeMember(userId) {
        this.members.delete(userId);
    }

    earnPoints(userId, purchaseAmount) {
        if (!this.members.has(userId)) {
            throw new Error('User  is not a member of this loyalty program.');
        }
        const pointsEarned = Math.floor(purchaseAmount * this.pointsPerPurchase);
        this.members.set(userId, this.members.get(userId) + pointsEarned);
        return pointsEarned;
    }

    getPoints(userId) {
        if (!this.members.has(userId)) {
            throw new Error('User  is not a member of this loyalty program.');
        }
        return this.members.get(userId);
    }

    redeemPoints(userId, points) {
        if (!this.members.has(userId)) {
            throw new Error('User  is not a member of this loyalty program.');
        }
        const currentPoints = this.members.get(userId);
        if (currentPoints < points) {
            throw new Error('Insufficient points to redeem.');
        }
        this.members.set(userId, currentPoints - points);
        return true; // Redemption successful
    }
}

module.exports = LoyaltyProgram;
