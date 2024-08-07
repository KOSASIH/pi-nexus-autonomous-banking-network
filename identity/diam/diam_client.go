package diam

import (
	"context"
	"fmt"
	"log"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/identity/diam/pb"
	"google.golang.org/grpc"
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
	return client.CreateIdentity(ctx, req)
}

func (c *DIAMClient) Authenticate(ctx context.Context, req *pb.AuthenticateRequest) (*pb.AuthenticateResponse, error) {
	client := pb.NewDIAMClient(c.conn)
	return client.Authenticate(ctx, req)
}

func (c *DIAMClient) GetIdentity(ctx context.Context, req *pb.GetIdentityRequest) (*pb.GetIdentityResponse, error) {
	client := pb.NewDIAMClient(c.conn)
	return client.GetIdentity(ctx, req)
}

func main() {
	client, err := NewDIAMClient("localhost:50052")
	if err != nil {
		log.Fatalf("failed to create DIAM client: %v", err)
	}
	defer client.conn.Close()

	req := &pb.CreateIdentityRequest{
		Name:      "John Doe",
		Email:     "johndoe@example.com",
		Password:  "password123",
		PublicKey: "some-public-key",
	}
	resp, err := client.CreateIdentity(context.Background(), req)
	if err != nil {
		log.Fatalf("failed to create identity: %v", err)
	}
	fmt.Println(resp)
}
