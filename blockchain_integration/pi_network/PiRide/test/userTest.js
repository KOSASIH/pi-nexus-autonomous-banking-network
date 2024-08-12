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
    await userContract.updateProfile(name, email, { from: user });
    const userProfile = await userContract.getUserProfile(user);
    assert.equal(userProfile.name, name);
    assert.equal(userProfile.email, email);
  });

  it("should allow users to request a ride", async () => {
    const user = accounts[1];
    const rideId = 1;
    await userContract.requestRide(rideId, { from: user });
    const rideRequest = await userContract.getRideRequest(rideId);
