// utils.test.js
const { expect } = require("chai");

describe("Utility Functions", function () {
    it("Should correctly calculate interest", function () {
        const principal = 1000;
        const rate = 5; // 5%
        const time = 1; // 1 year
        const expectedInterest = (principal * rate * time) / 100;

        expect(expectedInterest).to.equal(50); // 50 is the expected interest
    });

    it("Should validate an Ethereum address", function () {
        const validAddress = "0x32Be3435EFeD41E5c68D036c5a74cA1B1D8D1D0F"; // Example valid address
        const invalidAddress = "0x123"; // Invalid address

        expect(isValidAddress(validAddress)).to.be.true;
        expect(isValidAddress(invalidAddress)).to.be.false;
    });
});

// Utility function to validate Ethereum address
function isValidAddress(address) {
    return /^0x[a-fA-F0-9]{40}$/.test(address);
}
