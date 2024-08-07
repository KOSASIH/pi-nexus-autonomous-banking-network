// tests/integration/sssc_integration_test.go
package sssc_integration_test

import (
	"testing"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/sssc"
)

func TestProposeBlockIntegration(t *testing.T) {
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

	// Verify that the block was added to the node's blockchain
	if len(node.blockchain) != 1 {
		t.Errorf("blockchain length is not 1")
	}
	if node.blockchain[0].Hash != block.Hash {
		t.Errorf("block hash does not match")
	}
}

func TestVoteBlockIntegration(t *testing.T) {
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

	// Verify that the vote was added to the node's vote queue
	if len(node.votes) != 1 {
		t.Errorf("vote queue length is not 1")
	}
	if node.votes[0].Voter != vote.Voter {
		t.Errorf("vote voter does not match")
	}
}
