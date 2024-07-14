import React, { useState, useEffect } from "react";
import axios from "axios";

function FinancialAdvisory() {
  const [userInput, setUserInput] = useState("");
  const [advice, setAdvice] = useState("");

  useEffect(() => {
    const fetchAdvice = async () => {
      const response = await axios.post("/api/advice", { userInput });
      setAdvice(response.data);
    };
    fetchAdvice();
  }, [userInput]);

  const handleInputChange = (event) => {
    setUserInput(event.target.value);
  };

  return (
    <div>
      <h1>Financial Advisory</h1>
      <input type="text" value={userInput} onChange={handleInputChange} />
      <p>{advice}</p>
    </div>
  );
}

export default FinancialAdvisory;
