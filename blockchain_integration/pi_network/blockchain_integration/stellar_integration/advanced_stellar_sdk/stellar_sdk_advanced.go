package stellar

import (
	"context"
	"fmt"

	"github.com/stellar/go/clients/horizon"
	"github.com/stellar/go/keypair"
	"github.com/stellar/go/network"
	"github.com/stellar/go/txnbuild"
)

type AdvancedStellarSDK struct {
	horizonURL string
	networkPassphrase string
	client *horizon.Client
}

func NewAdvancedStellarSDK(horizonURL, networkPassphrase string) *AdvancedStellarSDK {
	return &AdvancedStellarSDK{
		horizonURL: horizonURL,
		networkPassphrase: networkPassphrase,
		client: horizon.NewClient(horizonURL),
	}
}

func (sdk *AdvancedStellarSDK) CreateAccount(seed, accountName string) (string, error) {
	kp, err := keypair.ParseSeed(seed)
	if err!= nil {
		return "", err
	}
	account, err := sdk.client.Account(kp.Address())
	if err!= nil {
		return "", err
	}
	if account == nil {
		tx, err := txnbuild.NewTransaction(
			kp.Address(),
			txnbuild.Sequence{Sequence: 1},
			txnbuild.Operation{
				Body: txnbuild.CreateAccount{
					Destination: kp.Address(),
					StartingBalance: "100.0",
				},
			},
		)
		if err!= nil {
			return "", err
		}
		tx, err = sdk.client.SubmitTransaction(context.Background(), tx)
		if err!= nil {
			return "", err
		}
		return tx.Hash, nil
	}
	return "", nil
}

func (sdk *AdvancedStellarSDK) IssueAsset(assetCode, assetIssuer string, amount string) error {
	tx, err := txnbuild.NewTransaction(
		assetIssuer,
		txnbuild.Sequence{Sequence: 1},
		txnbuild.Operation{
			Body: txnbuild.Payment{
				Destination: assetIssuer,
				Asset: txnbuild.CreditAsset{
					Code:   assetCode,
					Issuer: assetIssuer,
				},
				Amount: amount,
			},
		},
	)
	if err!= nil {
		return err
	}
	_, err = sdk.client.SubmitTransaction(context.Background(), tx)
	return err
}

func (sdk *AdvancedStellarSDK) CreateTrustline(sourceAccount, assetCode, assetIssuer string) error {
	tx, err := txnbuild.NewTransaction(
		sourceAccount,
		txnbuild.Sequence{Sequence: 1},
		txnbuild.Operation{
			Body: txnbuild.ChangeTrust{
				Asset: txnbuild.CreditAsset{
					Code:   assetCode,
					Issuer: assetIssuer,
				},
			},
		},
	)
	if err!= nil {
		return err
	}
	_, err = sdk.client.SubmitTransaction(context.Background(), tx)
	return err
}

func (sdk *AdvancedStellarSDK) PathPayment(sourceAccount, destinationAccount, assetCode, amount string) error {
	tx, err := txnbuild.NewTransaction(
		sourceAccount,
		txnbuild.Sequence{Sequence: 1},
		txnbuild.Operation{
			Body: txnbuild.PathPayment{
				SendAsset: txnbuild.CreditAsset{
					Code:   assetCode,
					Issuer: sourceAccount,
				},
				SendAmount: amount,
				Destination: destinationAccount,
				DestAsset: txnbuild.CreditAsset{
					Code:   assetCode,
					Issuer: destinationAccount,
				},
			},
		},
	)
	if err!= nil {
		return err
	}
	_, err = sdk.client.SubmitTransaction(context.Background(), tx)
	return err
}
