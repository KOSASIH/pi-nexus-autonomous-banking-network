package ntwk_test

import (
	"testing"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/ntwk"
)

func TestSendMessage(t *testing.T) {
	// Create a test Ntwk node
	node := ntwk.NewNtwkNode()

	// Create a test message
	message := &ntwk.Message{
		ID:   "message-1",
		Data: "some data",
	}

	// Send the message
	req := &ntwk.SendMessageRequest{
		Message: message,
	}
	resp, err := node.SendMessage(context.Background(), req)

	// Check the response
	if err != nil {
		t.Errorf("failed to send message: %v", err)
	}
	if resp.Result != "success" {
		t.Errorf("unexpected response: %v", resp)
	}
}

func TestGetMessage(t *testing.T) {
	// Create a test Ntwk node
	node := ntwk.NewNtwkNode()

	// Create a test message
	message := &ntwk.Message{
		ID:   "message-1",
		Data: "some data",
	}

	// Add the message to the node's message queue
	node.messages[message.ID] = message

	// Get the message
		req := &ntwk.GetMessageRequest{
		MessageID: message.ID,
	}
	resp, err := node.GetMessage(context.Background(), req)

	// Check the response
	if err != nil {
		t.Errorf("failed to get message: %v", err)
	}
	if resp.Message.ID != message.ID {
		t.Errorf("unexpected response: %v", resp)
	}
}
	
