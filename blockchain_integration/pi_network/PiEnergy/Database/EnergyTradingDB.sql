CREATE TABLE energy_producers (
    id SERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL
);

CREATE TABLE energy_consumers (
    id SERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL
);

CREATE TABLE energy_trades (
    id SERIAL PRIMARY KEY,
    producer_id INTEGER NOT NULL,
    consumer_id INTEGER NOT NULL,
    amount INTEGER NOT NULL,
    FOREIGN KEY (producer_id) REFERENCES energy_producers(id),
    FOREIGN KEY (consumer_id) REFERENCES energy_consumers(id)
);
