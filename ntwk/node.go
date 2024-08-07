package ntwk

import (
	"context"
	"fmt"
	"log"
	"net"
	"time"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/ntwk/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

type NtwkNode struct {
	pb.UnimplementedNtwkNodeServer
	nodeID string
	peers  map[string]*grpc.ClientConn
	messages map[string]*pb.Message
}

func (n *NtwkNode) SendMessage(ctx context.Context, req *pb.SendMessageRequest) (*pb.SendMessageResponse, error) {
	// Validate message
	if err := validateMessage(req.Message); err != nil {
		return nil, err
	}

	// Add message to local message queue
	n.messages[req.Message.ID] = req.Message

	// Broadcast message to peers
	for _, peer := range n.peers {
		client := pb.NewNtwkNodeClient(peer)
		_, err := client.SendMessage(ctx, req)
		if err != nil {
			log.Printf("failed to send message to peer %s: %v", peer.Target(), err)
		}
	}

	return &pb.SendMessageResponse{Result: "success"}, nil
}

func (n *NtwkNode) GetMessage(ctx context.Context, req *pb.GetMessageRequest) (*pb.GetMessageResponse, error) {
	// Get message from local message queue
	message, ok := n.messages[req.MessageID]
	if !ok {
		return nil, fmt.Errorf("message not found: %s", req.MessageID)
	}

	return &pb.GetMessageResponse{Message: message}, nil
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
		messages: make(map[string]*pb.Message),
	})

	reflection.Register(srv)

	log.Println("Ntwk node listening on port 50054")
	srv.Serve(lis)
}
