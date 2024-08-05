package smart_contract

import (
	"github.com/ethereum/go-ethereum/accounts/abi"
)

type PiSmartContract struct {
	//...
}

func (psc *PiSmartContract) DeployContract(contract []byte) string {
	//...
}

func (psc *PiSmartContract) ExecuteContractFunction(contract string, function string, args []byte) []byte {
	//...
}
