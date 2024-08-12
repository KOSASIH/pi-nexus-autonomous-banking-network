const RideContract = artifacts.require("RideContract");
const RideFactory = artifacts.require("RideFactory");
const PiRideToken = artifacts.require("PiRideToken");

contract("RideContract", accounts => {
  let rideContract;
  let rideFactory;
  let piRideToken;

  beforeEach(async () => {
    rideFactory = await RideFactory.deployed();
    piRideToken = await PiRideToken.deployed();
    rideContract = await rideFactory.createRideContract();
  });

  it("should create a new ride contract", async () => {
    assert.ok(rideContract.address);
  });

  it("should allow users to request a ride", async () => {
    const user = accounts[1];
    const rideId = 1;
    await rideContract.requestRide(rideId, { from: user });
    const rideRequest = await rideContract.getRideRequest(rideId);
    assert.equal(rideRequest.user, user);
  });

  it("should allow drivers to accept a ride request", async () => {
    const driver = accounts[2];
    const rideId = 1;
    await rideContract.acceptRideRequest(rideId, { from: driver });
    const rideRequest = await rideContract.getRideRequest(rideId);
    assert.equal(rideRequest.driver, driver);
  });

  it("should allow users to rate a ride", async () => {
    const user = accounts[1];
    const rideId = 1;
    const rating = 5;
    await rideContract.rateRide(rideId, rating, { from: user });
    const rideRating = await rideContract.getRideRating(rideId);
    assert.equal(rideRating, rating);
  });

  it("should transfer PiRide tokens to the driver", async () => {
    const driver = accounts[2];
    const rideId = 1;
    const amount = 10;
    await rideContract.completeRide(rideId, { from: driver });
    const driverBalance = await piRideToken.balanceOf(driver);
    assert.equal(driverBalance, amount);
  });
});
