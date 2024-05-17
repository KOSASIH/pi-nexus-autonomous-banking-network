/**
 * FabCar chaincode implementation.
 */
async function initializeCars(ctx) {
  /**
   * Initialize the car ledger with sample data.
   * @param {Context} ctx - The transaction context.
   */
  console.info("Initializing Car Ledger");
  const carData = [
    {
      make: "Toyota",
      model: "Prius",
      color: "blue",
      owner: "Tomoko",
    },
    {
      make: "Ford",
      model: "Mustang",
      color: "red",
      owner: "Brad",
    },
    {
      make: "Hyundai",
      model: "Tucson",
      color: "green",
      owner: "Jin Soo",
    },
    {
      make: "Volkswagen",
      model: "Passat",
      color: "yellow",
      owner: "Max",
    },
    {
      make: "Tesla",
      model: "S",
      color: "black",
      owner: "Adriana",
    },
    {
      make: "Peugeot",
      model: "205",
      color: "purple",
      owner: "Michel",
    },
    {
      make: "Chery",
      model: "S22L",
      color: "white",
      owner: "Aarav",
    },
    {
      make: "Fiat",
      model: "Punto",
      color: "violet",
      owner: "Pari",
    },
    {
      make: "Tata",
      model: "Nano",
      color: "indigo",
      owner: "Valeria",
    },
  ];

  for (let i = 0; i < carData.length; i++) {
    carData[i].docType = "car";
    try {
      await ctx.stub.putState(`${i}`, Buffer.from(JSON.stringify(carData[i])));
      console.info(
        `Added <--> ${carData[i].color} ${carData[i].make} ${carData[i].model}, owned by ${carData[i].owner}`,
      );
    } catch (error) {
      console.error(`Error adding car #${i}: ${error}`);
    }
  }
  console.info("Initialization complete");
}

async function queryCar(ctx, carNumber) {
  /**
   * Query a car from the ledger.
   * @param {Context} ctx - The transaction context.
   * @param {string} carNumber - The car number to query.
   * @returns {string} The car data as a JSON string.
   */
  try {
    const carAsBytes = await ctx.stub.getState(carNumber);
    if (!carAsBytes || carAsBytes.length === 0) {
      throw new Error(`Car #${carNumber} does not exist`);
    }
    console.log(`Car #${carNumber} is ${carAsBytes.toString()}`);
    return carAsBytes.toString();
  } catch (error) {
    console.error(`Error querying car #${carNumber}: ${error}`);
    throw error;
  }
}

async function updateCar(ctx, carNumber, carData) {
  /**
   * Update a car in the ledger.
   * @param {Context} ctx - The transaction context.
   * @param {string} carNumber - The car number to update.
   * @param {object} carData - The updated car data.
   */
  try {
    await ctx.stub.putState(carNumber, Buffer.from(JSON.stringify(carData)));
    console.info(`Updated car #${carNumber}`);
  } catch (error) {
    console.error(`Error updating car #${carNumber}: ${error}`);
    throw error;
  }
}

async function deleteCar(ctx, carNumber) {
  /**
   * Delete a car from the ledger.
   * @param {Context} ctx - The transaction context.
   * @param {string} carNumber - The car number to delete.
   */
  try {
    await ctx.stub.deleteState(carNumber);
    console.info(`Deleted car #${carNumber}`);
  } catch (error) {
    console.error(`Error deleting car #${carNumber}: ${error}`);
    throw error;
  }
}

module.exports = { initializeCars, queryCar, updateCar, deleteCar };
