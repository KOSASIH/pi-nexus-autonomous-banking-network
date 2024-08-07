package sssc

import (
	"context"
	"fmt"
	"log"
	"sync"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/consensus/sssc/pb"
	"google.golang.org/grpc"
)

type SSSCNode struct {
	pb.UnimplementedSSSCNodeServer
	shardID string
	nodes   map[string]*grpc.ClientConn
}

func (n *SSSCNode) JoinShard(ctx context.Context, req *pb.JoinShardRequest) (*pb.JoinShardResponse, error) {
	// TO DO: implement shard joining logic
	log.Printf("Received join shard request from %s", req.NodeID)
	n.nodes[req.NodeID] = req.Conn
	return &pb.JoinShardResponse{Result: "success"}, nil
}

func (n *SSSCNode) ProposeBlock(ctx context.Context, req *pb.ProposeBlockRequest) (*pb.ProposeBlockResponse, error) {
	// TO DO: implement block proposal logic
	log.Printf("Received propose block request from %s", req.NodeID)
	return &pb.ProposeBlockResponse{Result: "success"}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":50053")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterSSSCNodeServer(srv, &SSSCNode{shardID: "shard-1", nodes: make(map[string]*grpc.ClientConn)})

	log.Println("SSSC node listening on port 50053")
	srv.Serve(lis)
}
