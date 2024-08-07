package ntwk

import (
	"context"
	"fmt"
	"log"
	"net"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/network/ntwk/pb"
	"google.golang.org/grpc"
)

type NtwkNode struct {
	pb.UnimplementedNtwkNodeServer
	nodeID string
	peers  map[string]*grpc.ClientConn
}

func (n *NtwkNode) Connect(ctx context.Context, req *pb.ConnectRequest) (*pb.ConnectResponse, error) {
	// Establish connection to peer
	conn, err := grpc.Dial(req.NodeID, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	n.peers[req.NodeID] = conn

	return &pb.ConnectResponse{Result: "success"}, nil
}

func (n *NtwkNode) SendMessage(ctx context.Context, req *pb.SendMessageRequest) (*pb.SendMessageResponse, error) {
	// Send message to peer
	client := pb.NewNtwkNodeClient(n.peers[req.DestinationNodeID])
	_, err := client.SendMessage(ctx, req)
	if err != nil {
		return nil, err
	}

	return &pb.SendMessageResponse{Result: "success"}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":50054")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterNtwkNodeServer(srv, &NtwkNode{
		nodeID: "node-1",
		peers:  make(map[string]*grpc.ClientConn),
	})

	log.Println("Ntwk node listening on port 50054")
	srv.Serve(lis)
}
