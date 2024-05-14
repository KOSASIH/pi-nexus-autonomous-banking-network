package sharding

import (
	"fmt"
	"sync"
)

type ShardManager struct {
	Shards []*Shard
	Mutex  sync.Mutex
}

func NewShardManager(numShards int) *ShardManager {
	shards := make([]*Shard, numShards)
	for i := 0; i < numShards; i++ {
		shards[i] = NewShard(i)
	}
	return &ShardManager{
		Shards: shards,
	}
}

func (m *ShardManager) AddAccount(account string) {
	m.Mutex.Lock()
	defer m.Mutex.Unlock()
	shard := m.Shards[account[0]%uint8(len(m.Shards))]
	shard.AddAccount(account)
}

func (m *ShardManager) GetShard(account string) *Shard {
	m.Mutex.Lock()
	defer m.Mutex.Unlock()
	shard := m.Shards[account[0]%uint8(len(m.Shards))]
	for _, a := range shard.Account {
		if a == account {
			return shard
		}
	}
	return nil
}

func (m *ShardManager) PrintShards() {
	m.Mutex.Lock()
	defer m.Mutex.Unlock()
	for _, shard := range m.Shards {
		fmt.Printf("Shard %d: %v\n", shard.ID, shard.Account)
	}
}
