package security

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"io/ioutil"
	"math/big"
	"os"
	"time"
)

type endToEndEncryptionManager struct {
	keyDir string
}

func NewEndToEndEncryptionManager(keyDir string) *endToEndEncryptionManager {
	return &endToEndEncryptionManager{
		keyDir: keyDir,
	}
}

func (m *endToEndEncryptionManager) GenerateKey() error {
	keyPair, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return err
	}

	privateKeyBytes := x509.MarshalPKCS1PrivateKey(keyPair)
	privateKeyBlock := &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: privateKeyBytes,
	}

	privateKeyFile, err := os.Create(m.getKeyPath("private.pem"))
	if err != nil {
		return err
	}
	defer privateKeyFile.Close()

	if err = pem.Encode(privateKeyFile, privateKeyBlock); err != nil {
		return err
	}

	publicKeyBytes := x509.MarshalPKCS1PublicKey(&keyPair.PublicKey)
	publicKeyBlock := &pem.Block{
		Type:  "RSA PUBLIC KEY",
		Bytes: publicKeyBytes,
	}

	publicKeyFile, err := os.Create(m.getKeyPath("public.pem"))
	if err != nil {
		return err
	}
	defer publicKeyFile.Close()

	if err = pem.Encode(publicKeyFile, publicKeyBlock); err != nil {
		return err
	}

	return nil
}

func (m *endToEndEncryptionManager) GetKey(isPrivate bool) ([]byte, error) {
	var keyFile *os.File
	var err error

	if isPrivate {
		keyFile, err = os.Open(m.getKeyPath("private.pem"))
	} else {
		keyFile, err = os.Open(m.getKeyPath("public.pem"))
	}

	if err != nil {
		return nil, err
	}
	defer keyFile.Close()

	keyInfo, err := pem.Decode(keyFile)
	if err != nil {
		return nil, err
	}

	var key interface{}
	if isPrivate {
		key, err = x509.ParsePKCS1PrivateKey(keyInfo.Bytes)
	} else {
		key, err = x509.ParsePKCS1PublicKey(keyInfo.Bytes)
	}

	if err != nil {
		return nil, err
	}

	return x509.MarshalPKCS1Keyv1(isPrivate, key)
}

func (m *endToEndEncryptionManager) GetKeyPath(isPrivate bool) string {
	return m.getKeyPath(func() string {
		if isPrivate {
			return "private"
		}
		return "public"
	}())
}

func (m *endToEndEncryptionManager) getKeyPath(filename string) string {
	return m.keyDir + "/" + filename + ".pem"
}
