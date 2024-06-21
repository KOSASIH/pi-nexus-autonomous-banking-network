// credit_card_controller.go
package main

import (
	"encoding/json"
	"net/http"

	"github.com/go-kit/kit/endpoint"
	"github.com/go-validate/validate"
)

type CreditCardController struct{}

func (c *CreditCardController) GetCreditCard(w http.ResponseWriter, r *http.Request) {
	var request GetCreditCardRequest
	err := json.NewDecoder(r.Body).Decode(&request)
	if err!= nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	err = validate.NewValidator().Struct(request)
	if err!= nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// Implement credit card retrieval logic using the CreditCardService
}

func main() {
	controller := &CreditCardController{}
	http.Handle("/credit_card", MakeHttpHandler(controller.GetCreditCard, logger))
	log.Fatal(http.ListenAndServe(":8080", nil))
}
