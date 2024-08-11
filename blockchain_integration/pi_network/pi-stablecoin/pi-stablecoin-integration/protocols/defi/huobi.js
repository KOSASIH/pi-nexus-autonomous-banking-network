import axios from 'axios';

const huobiApiUrl = 'https://api.huobi.pro/v1';
const huobiApiKey = 'YOUR_API_KEY';
const huobiApiSecret = 'YOUR_API_SECRET';

export async function getHuobiBalance(address) {
  const params = {
    'account-id': address
  };
  const signature = await generateSignature(params);
  const headers = {
    'AccessKeyId': huobiApiKey,
    'Signature': signature,
    'Content-Type': 'application/json'
  };
  const response = await axios.get(`${huobiApiUrl}/account/accounts`, { headers });
  return response.data.data[0].balance;
}

export async function placeHuobiOrder(address, symbol, amount, price) {
  const params = {
    'account-id': address,
    'amount': amount,
    'price': price,
    'symbol': symbol,
    'type': 'limit'
  };
  const signature = await generateSignature(params);
  const headers = {
    'AccessKeyId': huobiApiKey,
    'Signature': signature,
    'Content-Type': 'application/json'
  };
  const response = await axios.post(`${huobiApiUrl}/orders`, params, { headers });
  return response.data.data;
}

async function generateSignature(params) {
  const paramString = Object.keys(params).sort().map(key => `${key}=${params[key]}`).join('&');
  const signature = crypto.createHmac('sha256', huobiApiSecret).update(paramString).digest('hex');
  return signature;
}
