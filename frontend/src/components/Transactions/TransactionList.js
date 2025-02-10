import React, { useEffect, useState } from 'react';
import { getUser Transactions } from '../../api';

const TransactionList = () => {
    const [transactions, setTransactions] = useState([]);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchTransactions = async () => {
            const token =localStorage.getItem('token');
            try {
                const response = await getUser Transactions(token);
                setTransactions(response.data.transactions);
            } catch (err) {
                setError(err.response.data.message);
            }
        };

        fetchTransactions();
    }, []);

    return (
        <div>
            <h2>Your Transactions</h2>
            {error && <p>{error}</p>}
            <ul>
                {transactions.map((transaction) => (
                    <li key={transaction.id}>
                        {transaction.description} - ${transaction.amount} on {new Date(transaction.date).toLocaleDateString()}
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default TransactionList;
