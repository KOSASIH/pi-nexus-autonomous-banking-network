package database

const (
	createUsersTable = `
CREATE TABLE IF NOT EXISTS users (
	id SERIAL PRIMARY KEY,
	name VARCHAR(255) NOT NULL UNIQUE,
	email VARCHAR(255) NOT NULL UNIQUE,
	password VARCHAR(255) NOT NULL,
	created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
`
	createAccountsTable = `
CREATE TABLE IF NOT EXISTS accounts (
	id SERIAL PRIMARY KEY,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	currency VARCHAR(3) NOT NULL,
	balance DECIMAL(18, 2) NOT NULL DEFAULT 0,
	created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
`
	createTransactionsTable = `
CREATE TABLE IF NOT EXISTS transactions (
	id SERIAL PRIMARY KEY,
	account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
	amount DECIMAL(18, 2) NOT NULL,
	type VARCHAR(10) NOT NULL CHECK (type IN ('deposit', 'withdrawal')),
	created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
`
)

func (db *DB) CreateSchema() error {
	_, err := db.Exec(createUsersTable)
	if err != nil {
		return err
	}
	_, err = db.Exec(createAccountsTable)
	if err != nil {
		return err
	}
	_, err = db.Exec(createTransactionsTable)
	if err != nil {
		return err
	}
	return nil
}
