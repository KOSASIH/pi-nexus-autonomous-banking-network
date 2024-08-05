module.exports = {
  async handleResponse(res, data, message) {
    res.json({ data, message });
  },

  async handleError(res, error, message) {
    res.status(500).json({ error, message });
  }
};
