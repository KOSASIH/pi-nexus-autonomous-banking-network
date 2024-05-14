package account

import (
	"errors"
	"sync"
)

type Account struct {
	ID       string
	Name     string
	Balance  float64
	Currency string
	Mutex    sync.Mutex
}

type AccountService struct {
	Accounts map[string]*Account
	Mutex    sync.Mutex
}

func NewAccountService() *AccountService {
	return &AccountService{
		Accounts: make(map[string]*Account),
	}
}

func (s *AccountService) CreateAccount(name string, currency string) (*Account, error) {
	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	id := GenerateAccountID()
	account := &Account{
		ID:     id,
		Name:   name,
		Balance: 0,
		Currency: currency,
	}
	s.Accounts[id] = account
	return account, nil
}

func (s *AccountService) GetAccount(id string) (*Account, error) {
	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	account, ok := s.Accounts[id]
	if !ok {
		return nil, errors.New("account not found")
	}
	return account, nil
}

func (s *AccountService) UpdateAccount(id string, name string, currency string) (*Account, error) {
	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	account, ok := s.Accounts[id]
	if !ok {
		return nil, errors.New("account not found")
	}
	account.Name = name
	account.Currency = currency
	return account, nil
}

func (s *AccountService) DeleteAccount(id string) error {
	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	delete(s.Accounts, id)
	return nil
}

func (s *AccountService) GetAccounts() []*Account {
	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	accounts := make([]*Account, 0, len(s.Accounts))
	for _, account := range s.Accounts {
		accounts = append(accounts, account)
	}
	return accounts
}

func (s *AccountService) Deposit(id string, amount float64) error {
	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	account, ok := s.Accounts[id]
	if !ok {
		return errors.New("account not found")
	}
	account.Balance += amount
	return nil
}

func (s *AccountService) Withdraw(id string, amount float64) error {
	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	account, ok := s.Accounts[id]
	if !ok {
		return errors.New("account not found")
	}
	if account.Balance < amount {
		return errors.New("insufficient balance")
	}
	account.Balance -= amount
	return nil
}

func GenerateAccountID() string {
	return "acc-" + RandString(10)
}

func RandString(n int) string {
	const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	result := make([]byte, n)
	for i := range result {
		result[i] = letters[rand.Intn(len(letters))]
	}
	return string(result)
}
