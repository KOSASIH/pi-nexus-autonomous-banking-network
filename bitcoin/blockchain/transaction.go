package main

type Transaction struct {
	Sender    string
	Recipient string
	Amount    float64
}

func NewTransaction(sender, recipient string, amount float64) *Transaction {
	return &Transaction{
		Sender:    sender,
		Recipient: recipient,
		Amount:    amount,
	}
}
