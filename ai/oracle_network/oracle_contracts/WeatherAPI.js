const axios = require('axios');

class WeatherAPI {
    async getData() {
        const response = await axios.get('https://api.openweathermap.org/data/2.5/weather?q=London,UK');
        return response.data.main.temp;
    }
}

module.exports = WeatherAPI;
