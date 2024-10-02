// Import the necessary libraries
const { RateLimiter } = require("limiter");

// Set up the rate limiter
const limiter = new RateLimiter({
  max: 100, // maximum requests per hour
  duration: 3600000, // 1 hour in milliseconds
});

// Implement rate limiting for Pi Network API requests
async function makeApiRequest(endpoint, params) {
  const limit = await limiter.get();
  if (limit.remaining === 0) {
    console.log("Rate limit exceeded. Waiting for 1 hour...");
    await new Promise((resolve) => setTimeout(resolve, 3600000));
  }

  try {
    const response = await piNetwork.api.request(endpoint, params);
    return response;
  } catch (error) {
    console.error("API request failed:", error);
  }
}

// Expose the rate-limited API request function
module.exports = {
  makeApiRequest,
};
