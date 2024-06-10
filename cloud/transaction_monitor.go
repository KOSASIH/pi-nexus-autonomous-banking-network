package main

import (
	"fmt"
	"log"
	"time"

	"github.com/gorilla/websocket"
)

type Transaction struct {
	ID        string    `json:"id"`
	Amount    float64  `json:"amount"`
	Timestamp time.Time `json:"timestamp"`
}

func main() {
	// Establish WebSocket connection
	conn, _, err := websocket.DefaultDialer.Dial("ws://localhost:8080/transactions", nil)
	if err!= nil {
		log.Fatal(err)
	}
	defer conn.Close()

	// Receive transactions in real-time
	for {
		_, message, err := conn.ReadMessage()
		if err!= nil {
			log.Fatal(err)
		}

		var transaction Transaction
		err = json.Unmarshal(message, &transaction)
		if err!= nil {
			log.Fatal(err)
		}

		fmt.Printf("Received transaction: %+v\n", transaction)
	}
}
