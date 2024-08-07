package mcip

import (
	"context"
	"fmt"
	"log"
	"sync"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/chain/mcip/pb"
	"google.golang.org/grpc"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
)

type MCIPService struct {
	pb.UnimplementedMCIPServer
	chainMap      map[string]*Chain
	chainLock     sync.RWMutex
	txPool        map[string][]*types.Transaction
	txPoolLock    sync.RWMutex
	blockchain    *Blockchain
	blockchainLock sync.RWMutex
}

type Chain struct {
	ID        string
	Name      string
	NetworkID uint64
	Genesis   *types.Block
}

type Blockchain struct {
	chainID string
	chain   *types.Blockchain
}

func (s *MCIPService) RegisterChain(ctx context.Context, req *pb.RegisterChainRequest) (*pb.RegisterChainResponse, error) {
	// Create new chain
	chain := &Chain{
		ID:        req.ChainID,
		Name:      req.ChainName,
		NetworkID: req.NetworkID,
		Genesis:   req.Genesis,
	}

	// Store chain
	s.chainLock.Lock()
	s.chainMap[req.ChainID] = chain
	s.chainLock.Unlock()

	return &pb.RegisterChainResponse{Result: "success"}, nil
}

func (s *MCIPService) GetChain(ctx context.Context, req *pb.GetChainRequest) (*pb.GetChainResponse, error) {
	// Retrieve chain from map
	s.chainLock.RLock()
	chain, ok := s.chainMap[req.ChainID]
	s.chainLock.RUnlock()
	if !ok {
		return nil, fmt.Errorf("chain not found")
	}

	return &pb.GetChainResponse{Chain: chain}, nil
}

func (s *MCIPService) CrossChainTransfer(ctx context.Context, req *pb.CrossChainTransferRequest) (*pb.CrossChainTransferResponse, error) {
	// Get source and destination chains
	s.chainLock.RLock()
	srcChain, ok := s.chainMap[req.SourceChainID]
	dstChain, ok := s.chainMap[req.DestinationChainID]
	s.chainLock.RUnlock()
	if !ok {
		return nil, fmt.Errorf("chain not found")
	}

	// Perform cross-chain transfer logic
	tx := &types.Transaction{
		From:     req.From,
		To:       req.To,
		Value:    req.Value,
		Gas:      req.Gas,
		GasPrice: req.GasPrice,
	}

	s.txPoolLock.Lock()
	s.txPool[req.SourceChainID] = append(s.txPool[req.SourceChainID], tx)
	s.txPoolLock.Unlock()

	return &pb.CrossChainTransferResponse{Result: "success"}, nil
}

func (s *MCIPService) MineBlock(ctx context.Context, req *pb.MineBlockRequest) (*pb.MineBlockResponse, error) {
	// Get chain
	s.chainLock.RLock()
	chain, ok := s.chainMap[req.ChainID]
	s.chainLock.RUnlock()
	if !ok {
		return nil, fmt.Errorf("chain not found")
	}

	// Mine block
	block, err := s.blockchain.MineBlock(chain, req.Transactions)
	if err != nil {
		return nil, err
	}

	return &pb.MineBlockResponse{Block: block}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterMCIPServer(srv, &MCIPService{
		chainMap:      make(map[string]*Chain),
		txPool:        make(map[string][]*types.Transaction),
		blockchain:    &Blockchain{},
	})

	log.Println("MCIP service listening on port 50051")
	srv.Serve(lis)
}
