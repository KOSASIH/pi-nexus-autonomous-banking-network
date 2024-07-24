import React, { useState } from 'react';
import { PiBrowser } from '@pi-network/pi-browser-sdk';

const PiBrowserUtilities = () => {
  const [unitConversion, setUnitConversion] = useState('');
  const [calculatorResult, setCalculatorResult] = useState('');
  const [qrCodeData, setQrCodeData] = useState('');
  const [currencyConversionRate, setCurrencyConversionRate] = useState({});

  const handleUnitConversion = async (fromUnit, toUnit, amount) => {
    // Convert units using Pi Browser's unit conversion API
    const converted = await PiBrowser.convertUnits(fromUnit, toUnit, amount);
    setUnitConversion(converted);
  };

  const handleCalculatorOperation = async (operation, num1, num2) => {
    // Perform complex mathematical operation using Pi Browser's calculator API
    const result = await PiBrowser.calculate(operation, num1, num2);
    setCalculatorResult(result);
  };

  const handleQrCodeGeneration = async (data) => {
    // Generate QR code using Pi Browser's QR code generator API
    const qrCode = await PiBrowser.generateQrCode(data);
    setQrCodeData(qrCode);
  };

  const handleCurrencyConversionRate = async () => {
    // Fetch current currency conversion rates using Pi Browser's API
    const rates = await PiBrowser.getCurrencyConversionRates();
    setCurrencyConversionRate(rates);
  };

  return (
    <div>
      <h1>Pi Browser Utilities</h1>
      <section>
        <h2>Unit Conversion</h2>
        <input
          type="text"
          value={unitConversion}
          onChange={e => handleUnitConversion(e.target.value, 'USD', 1)}
          placeholder="Enter amount to convert"
        />
        <select value="BTC" onChange={e => handleUnitConversion(e.target.value, 'USD', 1)}>
          <option value="BTC">Bitcoin</option>
          <option value="ETH">Ethereum</option>
          <option value="LTC">Litecoin</option>
        </select>
        <p>Converted amount: {unitConversion}</p>
      </section>
      <section>
        <h2>Calculator</h2>
        <input
          type="number"
          value={calculatorResult}
          onChange={e => handleCalculatorOperation('add', e.target.value, 2)}
          placeholder="Enter number to calculate"
        />
        <select value="add" onChange={e => handleCalculatorOperation(e.target.value, 2, 3)}>
          <option value="add">Add</option>
          <option value="subtract">Subtract</option>
          <option value="multiply">Multiply</option>
          <option value="divide">Divide</option>
        </select>
        <p>Result: {calculatorResult}</p>
      </section>
      <section>
        <h2>QR Code Generator</h2>
        <input
          type="text"
          value={qrCodeData}
          onChange={e => handleQrCodeGeneration(e.target.value)}
          placeholder="Enter data to generate QR code"
        />
        <img src={qrCodeData} alt="QR Code" />
      </section>
      <section>
        <h2>Currency Conversion Rates</h2>
        <button onClick={handleCurrencyConversionRate}>Fetch Rates</button>
        <ul>
          {Object.keys(currencyConversionRate).map((currency, index) => (
            <li key={index}>
              {currency}: {currencyConversionRate[currency]}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
};

export default PiBrowserUtilities;
