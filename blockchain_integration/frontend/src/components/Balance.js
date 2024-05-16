// frontend/src/components/Balance.js
import React, { useState, useEffect } from "react";
import axios from "axios";

const Balance = () => {
  const [balance, setBalance] = useState(0);
  const [address, setAddress] = useState("0x...");

  useEffect(() => {
    axios
      .get(`https://api.example.com/balance/${address}`)
      .then((response) => {
        setBalance(response.data.balance);
      })
      .catch((error) => {
        console.error(error);
      });
  }, [address]);

  return (
    <div>
      <h1>Balance: {balance}</h1>
      <input
        type="text"
        value={address}
        onChange={(e) => setAddress(e.target.value)}
      />
    </div>
  );
};

export default Balance;
