// sidra_identity_manager/main.go
package main

import (
	"context"
	"log"

	"google.golang.org/grpc"
	"golang.org/x/oauth2"
)

type IdentityManager struct{}

func (im *IdentityManager) Authenticate(ctx context.Context, req *AuthenticateRequest) (*AuthenticateResponse, error) {
	// Handle OAuth2 authentication flow
	config := &oauth2.Config{
		ClientID:     "sidra_identity_manager_client_id",
		ClientSecret: "sidra_identity_manager_client_secret",
		RedirectURL:  "http://sidra_identity_manager:8080/callback",
		Scopes:       []string{"openid", "profile", "email"},
		Endpoint: oauth2.Endpoint{
			AuthURL:  "https://sidra_identity_manager_auth_server.com/auth",
			TokenURL: "https://sidra_identity_manager_auth_server.com/token",
		},
	}

	code := req.GetCode()
	token, err := config.Exchange(ctx, code)
	if err != nil {
		return nil, err
	}

	// Verify user identity and return authentication response
	userID, err := verifyUserIdentity(token)
	if err != nil {
		return nil, err
	}

	return &AuthenticateResponse{UserId: userID}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":8080")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterIdentityManagerServer(srv, &IdentityManager{})

	log.Println("gRPC server listening on :8080")
	srv.Serve(lis)
}
