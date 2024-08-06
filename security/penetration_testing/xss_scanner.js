// xss_scanner.js
const axios = require("axios");

async function testXSS(url: string, payload: string) {
  try {
    const response = await axios.get(url, {
      params: { input: payload },
    });
    return response.data.includes(payload);
  } catch (error) {
    console.error(error);
    return false;
  }
}

async function main() {
  const url = "https://example.com/vulnerable_endpoint";
  const payloads = ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>"];

  for (const payload of payloads) {
    if (await testXSS(url, payload)) {
      console.log(`XSS vulnerability found: ${payload}`);
      break;
    }
  }
}

main();
