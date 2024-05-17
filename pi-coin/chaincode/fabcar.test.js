const { FileSystemWallet, Gateway } = require("fabric-network");
const path = require("path");
const fs = require("fs");

describe("FabCar chaincode", () => {
  const ccpPath = path.resolve(
    __dirname,
    "..",
    "..",
    "test",
    "fixtures",
    "connection.json",
  );
  const ccpJSON = fs.readFileSync(ccpPath, "utf8");
  const ccp = JSON.parse(ccpJSON);

  let wallet;
  let gateway;

  beforeAll(async () => {
    wallet = new FileSystemWallet("../identity/user1/wallet");
    gateway = new Gateway();
    await gateway.connect(ccp, {
      wallet,
      identity: "user1",
      discovery: { enabled: true, asLocalhost: true },
    });
    const network = await gateway.getNetwork("mychannel");
    const contract = network.getContract("fabcar");
    await contract.submitTransaction("initializeCars");
  });

  afterAll(async () => {
    gateway.disconnect();
  });

  test("should query a car", async () => {
    const carNumber = "0";
    const carString = await contract.evaluateTransaction("queryCar", carNumber);
    const car = JSON.parse(carString);
    expect(car.make).toEqual("Toyota");
    expect(car.model).toEqual("Prius");
    expect(car.color).toEqual("blue");
    expect(car.owner).toEqual("Tomoko");
  });

  test("should update a car", async () => {
    const carNumber = "0";
    const updatedCarData = {
      make: "Toyota",
      model: "Prius",
      color: "blue",
      owner: "Tomoko",
      newOwner: "John",
    };
    await contract.submitTransaction(
      "updateCar",
      carNumber,
      JSON.stringify(updatedCarData),
    );
    const carString = await contract.evaluateTransaction("queryCar", carNumber);
    const car = JSON.parse(carString);
    expect(car.newOwner).toEqual("John");
  });

  test("should delete a car", async () => {
    const carNumber = "0";
    await contract.submitTransaction("deleteCar", carNumber);
    const carString = await contract.evaluateTransaction("queryCar", carNumber);
    expect(carString).toEqual("");
  });
});
