package asset

import (
	"errors"
	"sync"
)

type Asset struct {
	ID       string
	Name     string
	Quantity float64
	Mutex    sync.Mutex
}

type AssetService struct {
	Assets map[string]*Asset
	Mutex  sync.Mutex
}

func NewAssetService() *AssetService {
	return &AssetService{
		Assets: make(map[string]*Asset),
	}
}

func (s *AssetService) CreateAsset(name string) (*Asset, error) {
	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	id := GenerateAssetID()
	asset := &Asset{
		ID:     id,
		Name:   name,
		Quantity: 0,
	}
	s.Assets[id] = asset
	return asset, nil
}

func (s *AssetService) GetAsset(id string) (*Asset, error) {
	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	asset, ok := s.Assets[id]
	if !ok {
		return nil, errors.New("asset not found")
	}
	return asset, nil
}

func (s *AssetService) UpdateAsset(id string, name string, quantity float64) (*Asset, error) {
	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	asset, ok := s.Assets[id]
	if !ok {
		return nil, errors.New("asset not found")
	}
	asset.Name = name
	asset.Quantity = quantity
	return asset, nil
}

func (s *AssetService) DeleteAsset(id string) error {
	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	delete(s.Assets, id)
	return nil
}

func (s *AssetService) GetAssets() []*Asset {
	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	assets := make([]*Asset, 0, len(s.Assets))
	for _, asset := range s.Assets {
		assets = append(assets, asset)
	}
	return assets
}

func GenerateAssetID() string {
	return "asset-" + RandString(10)
}

func RandString(n int) string {
	const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	result := make([]byte, n)
	for i := range result {
		result[i] = letters[rand.Intn(len(letters))]
	}
	return string(result)
}
