package diam_integration_test

import (
	"testing"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/diam"
)

func TestCreateIdentityIntegration(t *testing.T) {
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

	// Verify that the identity was added to the node's identity database
	if len(node.identities) != 1 {
		t.Errorf("identity database length is not 1")
	}
	if node.identities[identity.ID].Username != identity.Username {
		t.Errorf("identity username does not match")
	}
}

func TestAuthenticateIntegration(t *testing.T) {
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

	// Authenticate the identity
	req = &diam.AuthenticateRequest{
		Identity: identity,
	}
	resp, err = node.Authenticate(context.Background(), req)

	// Check the response
	if err != nil {
		t.Errorf("failed to authenticate: %v", err)
	}
		if !resp.Authenticated {
		t.Errorf("unexpected response: %v", resp)
	}

	// Verify that the authentication was successful
	if !node.IsAuthenticated(identity.ID) {
		t.Errorf("identity is not authenticated")
	}
}
