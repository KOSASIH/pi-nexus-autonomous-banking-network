package diam

import (
	"context"
	"fmt"
	"log"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/identity/diam/pb"
	"google.golang.org/grpc"
	"golang.org/x/crypto/bcrypt"
)

type DIAMClient struct {
	conn *grpc.ClientConn
}

func NewDIAMClient(addr string) (*DIAMClient, error) {
	conn, err := grpc.Dial(addr, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	return &DIAMClient{conn: conn}, nil
}

func (c *DIAMClient) CreateIdentity(ctx context.Context, req *pb.CreateIdentityRequest) (*pb.CreateIdentityResponse, error) {
	client := pb.NewDIAMClient(c.conn)
	hash, err := bcrypt.GenerateFromPassword([]byte(req.Password), 12)
	if err != nil {
		return nil, err
	}
	req.Password = string(hash)
	return client.CreateIdentity(ctx, req)
}

func (c *DIAMClient) Authenticate(ctx context.Context, req *pb.AuthenticateRequest) (*pb.AuthenticateResponse, error) {
	client := pb.NewDIAMClient(c.conn)
	err := bcrypt.CompareHashAndPassword([]byte(req.Password), []byte(req.Identity.Password))
	if err != nil {
		return nil, err
	}
	return client.Authenticate(ctx, req)
}

func (c *DIAMClient) Authenticate(ctx context.Context, req *pb.AuthenticateRequest) (*pb.AuthenticateResponse, error) {
	client := pb.NewDIAMClient(c.conn)
	err := bcrypt.CompareHashAndPassword([]byte(req.Password), []byte(req.Identity.Password))
	if err != nil {
		return nil, err
	}
	return client.Authenticate(ctx, req)
}

func main() {
	diamClient, err := NewDIAMClient("localhost:50052")
	if err != nil {
		log.Fatalf("failed to create DIAM client: %v", err)
	}
	defer diamClient.conn.Close()

	// Create identity
	req := &pb.CreateIdentityRequest{
		Identity: &pb.Identity{
			ID:       "user-1",
			Username: "john",
			Password: "password",
		},
	}
	resp, err := diamClient.CreateIdentity(context.Background(), req)
	if err != nil {
		log.Fatalf("failed to create identity: %v", err)
	}
	log.Println("Identity created:", resp.Identity)

	// Authenticate
	req = &pb.AuthenticateRequest{
		Identity: &pb.Identity{
			ID:       "user-1",
			Username: "john",
			Password: "password",
		},
	}
	resp, err = diamClient.Authenticate(context.Background(), req)
	if err != nil {
		log.Fatalf("failed to authenticate: %v", err)
	}
	log.Println("Authenticated:", resp.Authenticated)
}
