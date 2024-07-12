CREATE TABLE customers (
    id INT PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    country VARCHAR(255)
);

CREATE TABLE transactions (
    id INT PRIMARY KEY,
    customer_id INT,
    amount DECIMAL(10, 2),
    date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

SELECT c.name, SUM(t.amount) AS total_spent
FROM customers c
JOIN transactions t ON c.id = t.customer_id
GROUP BY c.name
ORDER BY total_spent DESC;
