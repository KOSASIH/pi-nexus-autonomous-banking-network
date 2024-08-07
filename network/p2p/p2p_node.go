package p2p

import (
	"context"
	"fmt"
	"log"
	"net"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/network/p2p/pb"
	"google.golang.org/grpc"
)

type P2PNode struct {
	pb.UnimplementedP2PNodeServer
	nodeID string
	peers  map[string]*grpc.ClientConn
}

func (n *P2PNode) Connect(ctx context.Context, req *pb.ConnectRequest) (*pb.ConnectResponse, error) {
	// TO DO: implement peer connection logic
	log.Printf("Received connect request from %s", req.NodeID)
	n.peers[req.NodeID] = req.Conn
	return &pb.ConnectResponse{Result: "success"}, nil
}

func (n *P2PNode) Disconnect(ctx context.Context, req *pb.DisconnectRequest) (*pb.DisconnectResponse, error) {
	// TO DO: implement peer disconnection logic
	log.Printf("Received disconnect request from %s", req.NodeID)
	delete(n.peers, req.NodeID)
	return &pb.DisconnectResponse{Result: "success"}, nil
}

func (n *P2PNode) SendMessage(ctx context.Context, req *pb.SendMessageRequest) (*pb.SendMessageResponse, error) {
	// TO DO: implement message sending logic
	log.Printf("Received send message request from %s", req.NodeID)
	return &pb.SendMessageResponse{Result: "success"}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":50054")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterP2PNodeServer(srv, &P2PNode{
		nodeID: "node-1",
		peers:  make(map[string]*grpc.ClientConn),
	})

	log.Println("P2P node listening on port 50054")
	srv.Serve(lis)
}
