// WeatherOracle.js
const axios = require("axios");
const { ethers } = require("ethers");
const DecentralizedOracleABI = require("./artifacts/DecentralizedOracle.json"); // Adjust the path as necessary

const provider = new ethers.providers.JsonRpcProvider("YOUR_INFURA_OR_ALCHEMY_URL");
const wallet = new ethers.Wallet("YOUR_PRIVATE_KEY", provider);
const oracleContractAddress = "YOUR_ORACLE_CONTRACT_ADDRESS";

async function fetchWeatherData(location) {
    // Simulate fetching weather data
    // In a real implementation, you would use an API like OpenWeatherMap
    const simulatedWeatherData = {
        temperature: Math.floor(Math.random() * 100), // Random temperature
        humidity: Math.floor(Math.random() * 100), // Random humidity
    };
    return simulatedWeatherData;
}

async function updateWeatherData(location) {
    const oracleContract = new ethers.Contract(oracleContractAddress, DecentralizedOracleABI.abi, wallet);
    const feedId = ethers.utils.keccak256(ethers.utils.toUtf8Bytes(location));

    const weatherData = await fetchWeatherData(location);
    const temperature = weatherData.temperature;

    try {
        const tx = await oracleContract.updateDataFeed(feedId, temperature);
        console.log(`Updating weather data... Transaction Hash: ${tx.hash}`);
        await tx.wait();
        console.log("Weather data updated successfully!");
    } catch (error) {
        console.error("Error updating weather data:", error);
    }
}

// Example usage
const location = "New York"; // Replace with the desired location
updateWeatherData(location);
