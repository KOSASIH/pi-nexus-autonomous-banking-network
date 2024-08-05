package node

import (
	"net"
	"sync"
)

type PiNode struct {
	//...
}

func (pn *PiNode) Start() {
	//...
}

func (pn *PiNode) ConnectToPeer(peer string) {
	//...
}

func (pn *PiNode) BroadcastBlock(block *Block) {
	//...
}

func (pn *PiNode) HandleIncomingBlock(block *Block) {
	//...
}
