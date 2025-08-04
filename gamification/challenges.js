// challenges.js
const axios = require('axios')

const CHALLENGES_API_URL = 'https://api.challenges.com'

const getChallenges = async () => {
  try {
    const response = await axios.get(`${CHALLENGES_API_URL}/challenges`)
    return response.data
  } catch (error) {
    console.error(`Error fetching challenges: ${error.message}`)
    throw error
  }
}

const submitChallengeResult = async (challengeId, userId, result) => {
  try {
    const response = await axios.post(`${CHALLENGES_API_URL}/results`, {
      challengeId,
      userId,
      result
    })
    return response.data
  } catch (error) {
    console.error(`Error submitting challenge result: ${error.message}`)
    throw error
  }
}

module.exports = {
  getChallenges,
  submitChallengeResult
}
