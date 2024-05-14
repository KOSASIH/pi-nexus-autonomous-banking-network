package caching

import (
	"fmt"
	"sync"
	"time"
)

type CacheManager struct {
	Caches map[string]*Cache
	Mutex  sync.Mutex
}

func NewCacheManager() *CacheManager {
	return &CacheManager{
		Caches: make(map[string]*Cache),
	}
}

func (m *CacheManager) GetCache(name string) (*Cache, bool) {
	m.Mutex.Lock()
	defer m.Mutex.Unlock()
	cache, ok := m.Caches[name]
	return cache, ok
}

func (m *CacheManager) SetCache(name string, cache *Cache) {
	m.Mutex.Lock()
	defer m.Mutex.Unlock()
	m.Caches[name] = cache
}

func (m *CacheManager) RemoveCache(name string) {
	m.Mutex.Lock()
	defer m.Mutex.Unlock()
	delete(m.Caches, name)
}

func (m *CacheManager) PrintCaches() {
	m.Mutex.Lock()
	defer m.Mutex.Unlock()
	for name, cache := range m.Caches {
		fmt.Printf("Cache %s:\n", name)
		fmt.Println(cache.Data)
		fmt.Println(cache.Expiry)
	}
}

func (m *CacheManager) CleanExpiredCaches() {
	m.Mutex.Lock()
	defer m.Mutex.Unlock()
	for name, cache := range m.Caches {
		for key, expiry := range cache.Expiry {
			if time.Now().After(expiry) {
				delete(cache.Data, key)
				delete(cache.Expiry, key)
			}
		}
	}
}
