package sssc

import (
	"context"
	"fmt"
	"log"
	"sync"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/consensus/sssc/pb"
	"google.golang.org/grpc"
	"github.com/dgraph-io/badger"
)

type SSSCNode struct {
	pb.UnimplementedSSSCNodeServer
	shardID string
	nodes   map[string]*grpc.ClientConn
	db      *badger.DB
	votes   map[string]map[string]bool
}

func (n *SSSCNode) ProposeBlock(ctx context.Context, req *pb.ProposeBlockRequest) (*pb.ProposeBlockResponse, error) {
	// Store proposed block in database
	err := n.db.Update(func(txn *badger.Txn) error {
		err := txn.Set([]byte("proposed_block_"+req.Block.Hash().Hex()), []byte(req.Block.Marshal()))
		return err
	})
	if err != nil {
		return nil, err
	}

	// Broadcast proposed block to other nodes
	for _, node := range n.nodes {
		client := pb.NewSSSCNodeClient(node)
		_, err := client.ProposeBlock(ctx, req)
		if err != nil {
			log.Printf("Error broadcasting proposed block to node %s: %v", node.Target(), err)
		}
	}

	return &pb.ProposeBlockResponse{Result: "success"}, nil
}

func (n *SSSCNode) VoteBlock(ctx context.Context, req *pb.VoteBlockRequest) (*pb.VoteBlockResponse, error) {
	// Store vote in database
	err := n.db.Update(func(txn *badger.Txn) error {
		err := txn.Set([]byte("vote_"+req.Block.Hash().Hex()+"_"+req.Vote), []byte("true"))
		return err
	})
	if err != nil {
		return nil, err
	}

	// Update vote count
	n.votes[req.Block.Hash().Hex()][req.Vote] = true

	// Check if block is committed
	if n.isCommitted(req.Block.Hash().Hex()) {
		// Commit block to blockchain
		err := n.commitBlock(req.Block)
		if err != nil {
			return nil, err
		}
	}

	return &pb.VoteBlockResponse{Result: "success"}, nil
}

func (n *SSSCNode) isCommitted(blockHash string) bool {
	// Check if block has been committed
	votes, ok := n.votes[blockHash]
	if !ok {
		return false
	}
	if len(votes) >= len(n.nodes)/2 {
		return true
	}
	return false
}

func (n *SSSCNode) commitBlock(block *types.Block) error {
	// Commit block to blockchain
	err := n.db.Update(func(txn *badger.Txn) error {
		err := txn.Set([]byte("block_"+block.Hash().Hex()), []byte(block.Marshal()))
		return err
	})
	if err != nil {
		return err
	}
	return nil
}

func main() {
	lis, err := net.Listen("tcp", ":50053")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	db, err := badger.Open("sssc.db")
	if err != nil {
		log.Fatalf("failed to open badger db: %v", err)
	}
	defer db.Close()

	srv := grpc.NewServer()
	pb.RegisterSSSCNodeServer(srv, &SSSCNode{
		shardID: "shard-1",
		nodes:   make(map[string]*grpc.ClientConn),
		db:      db,
		votes:   make(map[string]map[string]bool),
	})

	log.Println("SSSC node listening on port 50053")
	srv.Serve(lis)
}
