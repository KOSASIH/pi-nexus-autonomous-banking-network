package network

import (
	"net"
)

type PiNetwork struct {
	//...
}

func (pn *PiNetwork) ConnectToNode(node string) {
	//...
}

func (pn *PiNetwork) SendMessage(node string, message []byte) {
	//...
}

func (pn *PiNetwork) BroadcastMessage(message []byte) {
	//...
}
