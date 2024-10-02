// leaderboard.js
const axios = require('axios')

const LEADERBOARD_API_URL = 'https://api.leaderboard.com'

const getLeaderboard = async () => {
  try {
    const response = await axios.get(`${LEADERBOARD_API_URL}/leaderboard`)
    return response.data
  } catch (error) {
    console.error(`Error fetching leaderboard: ${error.message}`)
    throw error
  }
}

module.exports = {
  getLeaderboard
}
