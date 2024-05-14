// rewards_system.js
const axios = require('axios')
const { v4: uuidv4 } = require('uuid')

const REWARDS_API_URL = 'https://api.rewards.com'

const getUserRewards = async (userId) => {
  try {
    const response = await axios.get(
      `${REWARDS_API_URL}/users/${userId}/rewards`
    )
    return response.data
  } catch (error) {
    console.error(`Error fetching user rewards: ${error.message}`)
    throw error
  }
}

const addUserReward = async (userId, rewardType, amount) => {
  try {
    const rewardId = uuidv4()
    const response = await axios.post(`${REWARDS_API_URL}/rewards`, {
      userId,
      rewardType,
      amount,
      rewardId
    })
    return response.data
  } catch (error) {
    console.error(`Error adding user reward: ${error.message}`)
    throw error
  }
}

module.exports = {
  getUserRewards,
  addUserReward
}
