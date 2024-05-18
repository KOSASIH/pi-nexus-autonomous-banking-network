package main

import (
	"fmt"
	"math/big"
)

type Fine struct {
	Offender string
	Amount   *big.Int
}

type FineX struct {
	Fines      map[string]*big.Int
	TotalFines *big.Int
	TotalPaid  *big.Int
}

func NewFineX() *FineX {
	return &FineX{
		Fines:      make(map[string]*big.Int),
		TotalFines: big.NewInt(0),
		TotalPaid:  big.NewInt(0),
	}
}

func (fx *FineX) IssueFine(offender string, amount *big.Int) {
	fx.Fines[offender] = new(big.Int).Add(fx.Fines[offender], amount)
	fx.TotalFines.Add(fx.TotalFines, amount)
}

func (fx *FineX) PayFine(offender string, amount *big.Int) error {
	if _, ok := fx.Fines[offender]; !ok {
		return fmt.Errorf("offender %s has no fines to pay", offender)
	}
	if amount.Cmp(fx.Fines[offender]) > 0 {
		return fmt.Errorf("offender %s cannot pay more than their fine amount", offender)
	}
	fx.Fines[offender].Sub(fx.Fines[offender], amount)
	fx.TotalPaid.Add(fx.TotalPaid, amount)
	return nil
}

func (fx *FineX) GetFineBalance(offender string) (*big.Int, error) {
	if amount, ok := fx.Fines[offender]; ok {
		return amount, nil
	}
	return nil, fmt.Errorf("offender %s has no fine balance", offender)
}

func (fx *FineX) GetTotalFines() *big.Int {
	return fx.TotalFines
}

func (fx *FineX) GetTotalPaid() *big.Int {
	return fx.TotalPaid
}

func main() {
	fx := NewFineX()
	fx.IssueFine("Alice", big.NewInt(100))
	fx.IssueFine("Bob", big.NewInt(200))
	fmt.Println(fx.GetFineBalance("Alice")) // Output: 100
	fmt.Println(fx.GetFineBalance("Bob"))   // Output: 200
	fmt.Println(fx.GetTotalFines())         // Output: 300
	fx.PayFine("Alice", big.NewInt(50))
	fmt.Println(fx.GetFineBalance("Alice")) // Output: 50
	fmt.Println(fx.GetTotalPaid())          // Output: 50
}
