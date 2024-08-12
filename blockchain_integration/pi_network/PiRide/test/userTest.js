const UserContract = artifacts.require("UserContract");
const UserFactory = artifacts.require("UserFactory");
const PiRideToken = artifacts.require("PiRideToken");

contract("UserContract", accounts => {
  let userContract;
  let userFactory;
  let piRideToken;

  beforeEach(async () => {
    userFactory = await UserFactory.deployed();
    piRideToken = await PiRideToken.deployed();
    userContract = await userFactory.createUserContract();
  });

  it("should create a new user contract", async () => {
    assert.ok(userContract.address);
  });

  it("should allow users to update their profile", async () => {
    const user = accounts[1];
    const name = "John Doe";
    const email = "johndoe@example.com";
    const phoneNumber = "1234567890";
    await userContract.updateProfile(name, email, phoneNumber, { from: user });
    const userProfile = await userContract.getUserProfile(user);
    assert.equal(userProfile.name, name);
    assert.equal(userProfile.email, email);
    assert.equal(userProfile.phoneNumber, phoneNumber);
  });

  it("should allow users to add a payment method", async () => {
    const user = accounts[1];
    const paymentMethod = "credit card";
    await userContract.addPaymentMethod(paymentMethod, { from: user });
    const userPaymentMethods = await userContract.getUserPaymentMethods(user);
    assert.include(userPaymentMethods, paymentMethod);
  });

  it("should allow users to request a ride", async () => {
    const user = accounts[1];
    const rideId = 1;
    const pickupLocation = "123 Main St";
    const dropoffLocation = "456 Elm St";
    await userContract.requestRide(rideId, pickupLocation, dropoffLocation, { from: user });
    const rideRequest = await userContract.getRideRequest(rideId);
    assert.equal(rideRequest.user, user);
    assert.equal(rideRequest.pickupLocation, pickupLocation);
    assert.equal(rideRequest.dropoffLocation, dropoffLocation);
  });

  it("should allow users to cancel a ride request", async () => {
    const user = accounts[1];
    const rideId = 1;
    await userContract.requestRide(rideId, "123 Main St", "456 Elm St", { from: user });
    await userContract.cancelRideRequest(rideId, { from: user });
    const rideRequest = await userContract.getRideRequest(rideId);
    assert.isNull(rideRequest);
  });

  it("should allow users to rate a ride", async () => {
    const user = accounts[1];
    const rideId = 1;
    const rating = 5;
    await userContract.rateRide(rideId, rating, { from: user });
    const rideRating = await userContract.getRideRating(rideId);
    assert.equal(rideRating, rating);
  });

  it("should transfer PiRide tokens to the user", async () => {
    const user = accounts[1];
    const amount = 10;
    await userContract.transferTokens(user, amount);
    const userBalance = await piRideToken.balanceOf(user);
    assert.equal(userBalance, amount);
  });
});
