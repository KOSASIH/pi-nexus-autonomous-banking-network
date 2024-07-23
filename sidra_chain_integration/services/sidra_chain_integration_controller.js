const express = require("express");
const app = express();
const bodyParser = require("body-parser");
const axios = require("axios");

app.use(bodyParser.json());

app.post("/sidra_chain_integration", async (req, res) => {
  const { chain_id, chain_name } = req.body;
  try {
    const response = await axios.post("http://localhost:5000/sidra_chain", {
      chain_id,
      chain_name,
    });
    res.json(response.data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Error creating Sidra Chain" });
  }
});

app.listen(3000, () => {
  console.log("Server listening on port 3000");
});
