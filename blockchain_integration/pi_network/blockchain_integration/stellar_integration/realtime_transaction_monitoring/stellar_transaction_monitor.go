package stellar

import (
	"context"
	"fmt"
	"log"

	"github.com/gorilla/websocket"
	"github.com/stellar/go/clients/horizon"
)

type StellarTransactionMonitor struct {
	horizonURL string
	networkPassphrase string
	client *horizon.Client
	ws *websocket.Conn
}

func NewStellarTransactionMonitor(horizonURL, networkPassphrase string) *StellarTransactionMonitor {
	return &StellarTransactionMonitor{
		horizonURL: horizonURL,
		networkPassphrase: networkPassphrase,
		client: horizon.NewClient(horizonURL),
	}
}

func (monitor *StellarTransactionMonitor) StartMonitoring() error {
	var err error
	monitor.ws, _, err = websocket.DefaultDialer.Dial(fmt.Sprintf("wss://%s/websocket", monitor.horizonURL), nil)
	if err!= nil {
		return err
	}
	err = monitor.ws.WriteMessage(websocket.TextMessage, []byte(`{"command": "listen", "stream": "transactions"}`))
	if err!= nil {
		return err
	}
	go monitor.listen()
	return nil
}

func (monitor *StellarTransactionMonitor) listen() {
	for {
		_, message, err := monitor.ws.ReadMessage()
		if err!= nil {
			log.Println(err)
			return
		}
		tx, err := txnbuild.TransactionFromXDR(message)
		if err!= nil {
			log.Println(err)
			continue
		}
		log.Println(tx.Hash)
	}
}

func (monitor *StellarTransactionMonitor) StopMonitoring() error {
	err := monitor.ws.WriteMessage(websocket.TextMessage, []byte(`{"command": "unsubscribe", "stream": "transactions"}`))
	if err!= nil {
		return err
	}
	err = monitor.ws.Close()
	return err
}
