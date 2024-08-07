package main

import (
	"context"
	"fmt"
	"log"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/sssc/pb"
	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/diam/pb"
	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/ntwk/pb"
	"google.golang.org/grpc"
)

func main() {
	// Create gRPC clients for each microservice
	ssscClient, err := grpc.Dial("localhost:50053", grpc.WithInsecure())
	if err != nil {
		log.Fatalf("failed to dial sssc: %v", err)
	}
	defer ssscClient.Close()

	diamClient, err := grpc.Dial("localhost:50052", grpc.WithInsecure())
	if err != nil {
		log.Fatalf("failed to dial diam: %v", err)
	}
	defer diamClient.Close()

	ntwkClient, err := grpc.Dial("localhost:50054", grpc.WithInsecure())
	if err != nil {
		log.Fatalf("failed to dial ntwk: %v", err)
	}
	defer ntwkClient.Close()

	// Create CLI commands
	fmt.Println("Pi Nexus Autonomous Banking Network CLI")
	fmt.Println("----------------------------------------")

	for {
		fmt.Print("Enter command: ")
		var cmd string
		fmt.Scanln(&cmd)

		switch cmd {
		case "create-identity":
			// Create identity using diam microservice
			req := &diam.CreateIdentityRequest{
				Identity: &diam.Identity{
					ID:       "user-1",
					Username: "john",
					Password: "password",
				},
			}
			resp, err := diam.NewDIAMClient(diamClient).CreateIdentity(context.Background(), req)
			if err != nil {
				log.Fatalf("failed to create identity: %v", err)
			}
			fmt.Println("Identity created:", resp.Identity)

		case "authenticate":
			// Authenticate using diam microservice
			req := &diam.AuthenticateRequest{
				Identity: &diam.Identity{
					ID:       "user-1",
					Username: "john",
					Password: "password",
				},
			}
			resp, err := diam.NewDIAMClient(diamClient).Authenticate(context.Background(), req)
			if err != nil {
				log.Fatalf("failed to authenticate: %v", err)
			}
			fmt.Println("Authenticated:", resp. Identity)
      		case "propose-block":
			// Propose block using sssc microservice
			req := &sssc.ProposeBlockRequest{
				Block: &sssc.Block{
					Hash: "block-1",
					Data: "some data",
				},
			}
			resp, err := sssc.NewSSSCNodeClient(ssscClient).ProposeBlock(context.Background(), req)
			if err != nil {
				log.Fatalf("failed to propose block: %v", err)
			}
			fmt.Println("Block proposed:", resp.Result)

		case "vote-block":
			// Vote on block using sssc microservice
			req := &sssc.VoteBlockRequest{
				Block: &sssc.Block{
					Hash: "block-1",
				},
				Vote: &sssc.Vote{
					Voter: "node-1",
					Value: true,
				},
			}
			resp, err := sssc.NewSSSCNodeClient(ssscClient).VoteBlock(context.Background(), req)
			if err != nil {
				log.Fatalf("failed to vote on block: %v", err)
			}
			fmt.Println("Voted on block:", resp.Result)

		case "send-message":
			// Send message using ntwk microservice
			req := &ntwk.SendMessageRequest{
				Message: &ntwk.Message{
					ID:   "message-1",
					Data: "some data",
				},
			}
			resp, err := ntwk.NewNtwkNodeClient(ntwkClient).SendMessage(context.Background(), req)
			if err != nil {
				log.Fatalf("failed to send message: %v", err)
			}
			fmt.Println("Message sent:", resp.Result)

		case "get-message":
			// Get message using ntwk microservice
			req := &ntwk.GetMessageRequest{
				MessageID: "message-1",
			}
			resp, err := ntwk.NewNtwkNodeClient(ntwkClient).GetMessage(context.Background(), req)
			if err != nil {
				log.Fatalf("failed to get message: %v", err)
			}
			fmt.Println("Message:", resp.Message)

		default:
			fmt.Println("Invalid command")
		}
	}
}
