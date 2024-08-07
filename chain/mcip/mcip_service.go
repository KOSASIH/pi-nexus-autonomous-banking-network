package mcip

import (
	"context"
	"fmt"
	"log"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/chain/mcip/pb"
	"google.golang.org/grpc"
)

type MCIPService struct {
	pb.UnimplementedMCIPServer
}

func (s *MCIPService) RequestInteroperability(ctx context.Context, req *pb.InteroperabilityRequest) (*pb.InteroperabilityResponse, error) {
	// TO DO: implement interoperability logic
	log.Printf("Received interoperability request from %s to %s", req.SourceChain, req.TargetChain)
	return &pb.InteroperabilityResponse{Result: "success"}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterMCIPServer(srv, &MCIPService{})

	log.Println("MCIP service listening on port 50051")
	srv.Serve(lis)
}
