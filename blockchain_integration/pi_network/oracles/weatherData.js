// oracles/weatherData.js

const axios = require("axios");

const API_KEY = "YOUR_WEATHER_API_KEY"; // Replace with your weather API key
const BASE_URL = "https://api.weatherapi.com/v1/current.json"; // Example weather API endpoint

class WeatherData {
    constructor() {}

    async getCurrentWeather(location) {
        try {
            const response = await axios.get(`${BASE_URL}?key=${API_KEY}&q=${location}`);
            const weatherData = response.data;
            return {
                location: weatherData.location.name,
                temperature: weatherData.current.temp_c,
                condition: weatherData.current.condition.text,
                humidity: weatherData.current.humidity,
                wind: weatherData.current.wind_kph,
            };
        } catch (error) {
            console.error("Error fetching weather data:", error);
            throw error;
        }
    }
}

module.exports = WeatherData;
