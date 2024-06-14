import React, { useState, useEffect } from 'eact';
import axios from 'axios';

const Budgeting = () => {
  const [budget, setBudget] = useState({});
  const [expenses, setExpenses] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('/api/budget')
     .then(response => {
        setBudget(response.data);
        setLoading(false);
      })
     .catch(error => {
        console.error(error);
      });

    axios.get('/api/expenses')
     .then(response => {
        setExpenses(response.data);
      })
     .catch(error => {
        console.error(error);
      });
  }, []);

  const handleAddExpense = (expense) => {
    axios.post('/api/expenses', expense)
     .then(response => {
        setExpenses((prevExpenses) => [...prevExpenses, response.data]);
      })
     .catch(error => {
        console.error(error);
      });
  };

  return (
    <div>
      {loading? (
        <p>Loading...</p>
      ) : (
        <div>
          <h2>Budget: {budget.amount} {budget.currency}</h2>
          <ul>
            {expenses.map((expense) => (
              <li key={expense.id}>
                {expense.amount} {expense.currency} - {expense.description}
              </li>
            ))}
          </ul>
          <button onClick={() => handleAddExpense({ amount: 10, currency: 'USD', description: 'Test Expense' })}>
            Add Expense
          </button>
        </div>
      )}
    </div>
  );
};

export default Budgeting;
