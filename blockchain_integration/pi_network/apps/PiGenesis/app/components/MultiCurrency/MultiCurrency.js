import React, { useState, useEffect } from 'eact';
import axios from 'axios';

const MultiCurrency = () => {
  const [currencies, setCurrencies] = useState([]);
  const [exchangeRates, setExchangeRates] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('/api/currencies')
     .then(response => {
        setCurrencies(response.data);
      })
     .catch(error => {
        console.error(error);
      });

    axios.get('/api/exchange-rates')
     .then(response => {
        setExchangeRates(response.data);
        setLoading(false);
      })
     .catch(error => {
        console.error(error);
      });
  }, []);

  const handleConvertCurrency = (amount, fromCurrency, toCurrency) => {
    const conversionRate = exchangeRates[fromCurrency] / exchangeRates[toCurrency];
    const convertedAmount = amount * conversionRate;

    return convertedAmount;
  };

  return (
    <div>
      {loading? (
        <p>Loading...</p>
      ) : (
        <div>
          <h2>Currencies:</h2>
          <ul>
            {currencies.map((currency) => (
              <li key={currency.code}>
                {currency.code}: {currency.name}
              </li>
            ))}
          </ul>
          <h2>Exchange Rates:</h2>
          <ul>
            {Object.entries(exchangeRates).map(([currencyCode, rate]) => (
              <li key={currencyCode}>
                {currencyCode}: {rate}
              </li>
            ))}
          </ul>
          <h2>Currency Conversion:</h2>
          <p>
            Enter an amount and select a currency to convert:
            <input type="number" />
            <select>
              {currencies.map((currency) => (
                <option key={currency.code} value={currency.code}>
                  {currency.code}
                </option>
              ))}
            </select>
            <button>Convert</button>
          </p>
          <p>Converted Amount: {handleConvertCurrency(10, 'USD', 'EUR')} EUR</p>
        </div>
      )}
    </div>
  );
};

export default MultiCurrency;
