package caching

import (
	"sync"
	"time"
)

type Cache struct {
	Data     map[string]interface{}
	Expiry   map[string]time.Time
	Mutex    sync.Mutex
	Capacity int
}

func NewCache(capacity int) *Cache {
	return &Cache{
		Data:     make(map[string]interface{}),
		Expiry:   make(map[string]time.Time),
		Capacity: capacity,
	}
}

func (c *Cache) Get(key string) (interface{}, bool) {
	c.Mutex.Lock()
	defer c.Mutex.Unlock()
	val, ok := c.Data[key]
	if !ok {
		return nil, false
	}
	if time.Now().After(c.Expiry[key]) {
		delete(c.Data, key)
		delete(c.Expiry, key)
		return nil, false
	}
	return val, true
}

func (c *Cache) Set(key string, val interface{}, expiry time.Duration) {
	c.Mutex.Lock()
	defer c.Mutex.Unlock()
	if len(c.Data) >= c.Capacity {
		for k := range c.Data {
			delete(c.Data, k)
			delete(c.Expiry, k)
			break
		}
	}
	c.Data[key] = val
	c.Expiry[key] = time.Now().Add(expiry)
}
