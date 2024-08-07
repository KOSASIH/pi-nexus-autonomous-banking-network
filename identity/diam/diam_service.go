package diam

import (
	"context"
	"fmt"
	"log"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/identity/diam/pb"
	"google.golang.org/grpc"
	"golang.org/x/crypto/bcrypt"
)

type DIAMService struct {
	pb.UnimplementedDIAMServer
	identityStore map[string]*Identity
}

type Identity struct {
	ID        string
	Name      string
	Email     string
	Password  string
	PublicKey string
}

func (s *DIAMService) CreateIdentity(ctx context.Context, req *pb.CreateIdentityRequest) (*pb.CreateIdentityResponse, error) {
	// Hash password
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(req.Password), 12)
	if err != nil {
		return nil, err
	}

	// Create new identity
	identity := &Identity{
		ID:        req.Name,
		Name:      req.Name,
		Email:     req.Email,
		Password:  string(hashedPassword),
		PublicKey: req.PublicKey,
	}

	// Store identity
	s.identityStore[req.Name] = identity

	return &pb.CreateIdentityResponse{Identity: identity.ID}, nil
}

func (s *DIAMService) Authenticate(ctx context.Context, req *pb.AuthenticateRequest) (*pb.AuthenticateResponse, error) {
	// Retrieve identity from store
	identity, ok := s.identityStore[req.IdentityID]
	if !ok {
		return nil, fmt.Errorf("identity not found")
	}

	// Verify password
	err := bcrypt.CompareHashAndPassword([]byte(identity.Password), []byte(req.Password))
	if err != nil {
		return nil, err
	}

	// Return authentication token
	token := "some-auth-token"
	return &pb.AuthenticateResponse{Token: token}, nil
}

func (s *DIAMService) GetIdentity(ctx context.Context, req *pb.GetIdentityRequest) (*pb.GetIdentityResponse, error) {
	// Retrieve identity from store
	identity, ok := s.identityStore[req.IdentityID]
	if !ok {
		return nil, fmt.Errorf("identity not found")
	}

	return &pb.GetIdentityResponse{Identity: identity}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":50052")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterDIAMServer(srv, &DIAMService{identityStore: make(map[string]*Identity)})

	log.Println("DIAM service listening on port 50052")
	srv.Serve(lis)
}
