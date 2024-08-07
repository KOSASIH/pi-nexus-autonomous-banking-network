package mcip

import (
	"context"
	"fmt"
	"log"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/chain/mcip/pb"
	"google.golang.org/grpc"
)

func main() {
	conn, err := grpc.Dial(":50051", grpc.WithInsecure())
	if err != nil {
		log.Fatalf("failed to dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewMCIPClient(conn)

	req := &pb.InteroperabilityRequest{
		SourceChain: "Pi Network",
		TargetChain:  "Ethereum",
		Payload:      []byte("Hello, Ethereum!"),
	}

	resp, err := client.RequestInteroperability(context.Background(), req)
	if err != nil {
		log.Fatalf("failed to request interoperability: %v", err)
	}

	log.Printf("Received response from MCIP service: %s", resp.Result)
}
