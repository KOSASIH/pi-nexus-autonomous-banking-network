// tests/unit/sssc_test.go
package sssc_test

import (
	"testing"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/sssc"
)

func TestProposeBlock(t *testing.T) {
	// Create a test SSSC node
	node := sssc.NewSSSCNode()

	// Create a test block
	block := &sssc.Block{
		Hash: "block-1",
		Data: "some data",
	}

	// Propose the block
	req := &sssc.ProposeBlockRequest{
		Block: block,
	}
	resp, err := node.ProposeBlock(context.Background(), req)

	// Check the response
	if err != nil {
		t.Errorf("failed to propose block: %v", err)
	}
	if resp.Result != "success" {
		t.Errorf("unexpected response: %v", resp)
	}
}

func TestVoteBlock(t *testing.T) {
	// Create a test SSSC node
	node := sssc.NewSSSCNode()

	// Create a test block
	block := &sssc.Block{
		Hash: "block-1",
	}

	// Create a test vote
	vote := &sssc.Vote{
		Voter: "node-1",
		Value: true,
	}

	// Vote on the block
	req := &sssc.VoteBlockRequest{
		Block: block,
		Vote:  vote,
	}
	resp, err := node.VoteBlock(context.Background(), req)

	// Check the response
	if err != nil {
		t.Errorf("failed to vote on block: %v", err)
	}
	if resp.Result != "success" {
		t.Errorf("unexpected response: %v", resp)
	}
}
