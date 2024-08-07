package diam_test

import (
	"testing"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/diam"
)

func TestCreateIdentity(t *testing.T) {
	// Create a test DIAM node
	node := diam.NewDIAMNode()

	// Create a test identity
	identity := &diam.Identity{
		ID:       "user-1",
		Username: "john",
		Password: "password",
	}

	// Create the identity
	req := &diam.CreateIdentityRequest{
		Identity: identity,
	}
	resp, err := node.CreateIdentity(context.Background(), req)

	// Check the response
	if err != nil {
		t.Errorf("failed to create identity: %v", err)
	}
	if resp.Identity.ID != identity.ID {
		t.Errorf("unexpected response: %v", resp)
	}
}

func TestAuthenticate(t *testing.T) {
	// Create a test DIAM node
	node := diam.NewDIAMNode()

	// Create a test identity
	identity := &diam.Identity{
		ID:       "user-1",
		Username: "john",
		Password: "password",
	}

	// Authenticate the identity
	req := &diam.AuthenticateRequest{
		Identity: identity,
	}
	resp, err := node.Authenticate(context.Background(), req)

	// Check the response
	if err != nil {
		t.Errorf("failed to authenticate: %v", err)
	}
	if !resp.Authenticated {
		t.Errorf("unexpected response: %v", resp)
	}
}
