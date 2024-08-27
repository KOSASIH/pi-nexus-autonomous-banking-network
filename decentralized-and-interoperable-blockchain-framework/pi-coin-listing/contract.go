package pi_coin_listing

import (
	"fmt"

	"github.com/ethereum/go-ethereum/accounts/abi/bind"
	"github.com/ethereum/go-ethereum/common"
)

// PiCoinListingContract is a custom contract implementation for Pi Coin listing
type PiCoinListingContract struct {
	*bind.Contract
}

func NewPiCoinListingContract(address common.Address) (*PiCoinListingContract, error) {
	contract, err := bind.NewContract(address, PiCoinListingABI, nil, nil)
	if err != nil {
		return nil, err
	}
	return &PiCoinListingContract{contract}, nil
}

func (c *PiCoinListingContract) ListPiCoin(piCoinAddress common.Address) error {
	tx, err := c.ListPiCoinTx(piCoinAddress)
	if err != nil {
		return err
	}
	_, err = c.CallTx(tx)
	return err
}

func (c *PiCoinListingContract) ListPiCoinTx(piCoinAddress common.Address) (*types.Transaction, error) {
	return c.CallTx(&types.Transaction{
		To:   c.Address,
		Data: []byte{0x01}, // ListPiCoin function selector
		Value: big.NewInt(0),
		Gas:   20000,
		GasPrice: big.NewInt(20),
	})
}
