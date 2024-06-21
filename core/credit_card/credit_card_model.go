// credit_card_model.go
package main

import (
	"gorm.io/gorm"
	"github.com/go-validate/validate"
	"golang.org/x/crypto/nacl/secretbox"
)

type CreditCard struct {
	gorm.Model
	Number     string `gorm:"uniqueIndex" validate:"required,credit_card_number"`
	Expiration string `validate:"required,expiration_date"`
	CVV        string `validate:"required,cvv"`
}

func (c *CreditCard) Validate() error {
	return validate.NewValidator().Struct(c)
}

func (c *CreditCard) Encrypt() error {
	// Implement credit card encryption logic using NaCl
}
