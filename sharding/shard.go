package sharding

import (
	"math/rand"
	"time"
)

type Shard struct {
	ID      int
	Account []string
}

func NewShard(id int) *Shard {
	return &Shard{
		ID: id,
	}
}

func (s *Shard) AddAccount(account string) {
	s.Account = append(s.Account, account)
}

func NewRandomShard(accounts []string) *Shard {
	rand.Seed(time.Now().UnixNano())
	id := rand.Intn(100)
	shard := NewShard(id)
	for _, account := range accounts {
		shard.AddAccount(account)
	}
	return shard
}

func ShardAccounts(accounts []string, numShards int) []*Shard {
	rand.Seed(time.Now().UnixNano())
	shards := make([]*Shard, numShards)
	for i := 0; i < numShards; i++ {
		shards[i] = NewShard(i)
	}
	for _, account := range accounts {
		shardID := rand.Intn(numShards)
		shards[shardID].AddAccount(account)
	}
	return shards
}
