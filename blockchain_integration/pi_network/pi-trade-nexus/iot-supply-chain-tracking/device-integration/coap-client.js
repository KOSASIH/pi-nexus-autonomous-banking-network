// Import required libraries
const coap = require('coap');
const config = require('../config');

// Set up CoAP client
const client = coap.request({
  host: config.coapServerUrl,
  port: config.coapServerPort,
  pathname: '/iot/supply-chain/tracking',
});

// Define CoAP methods
const methods = {
  GET: 'get',
  POST: 'post',
  PUT: 'put',
  DELETE: 'delete',
};

// Handle CoAP requests
client.on('response', (res) => {
  console.log(`Received CoAP response: ${res.code} ${res.method}`);

  // Handle CoAP response data
  res.on('data', (data) => {
    console.log(`Received CoAP data: ${data.toString()}`);
    // Process CoAP response data
  });
});

// Send CoAP requests
const sendCoapRequest = (method, data) => {
  client.write({
    method: methods[method],
    payload: data,
  });
};

// Export CoAP client function
module.exports = {
  sendCoapRequest,
};
