package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
)

type Cybersecurity struct {
	rules []string
}

func NewCybersecurity() *Cybersecurity {
	return &Cybersecurity{rules: []string{}}
}

func (c *Cybersecurity) LoadRules(filename string) error {
	data, err := ioutil.ReadFile(filename)
	if err!= nil {
		return err
	}
	c.rules = strings.Split(string(data), "\n")
	return nil
}

func (c *Cybersecurity) DetectThreats(data []byte) []string {
	threats := []string{}
	for _, rule := range c.rules {
		hash := sha256.Sum256(data)
		hashStr := hex.EncodeToString(hash[:])
		if strings.Contains(hashStr, rule) {
			threats = append(threats, rule)
		}
	}
	return threats
}

func main() {
	cybersecurity := NewCybersecurity()
	err := cybersecurity.LoadRules("rules.txt")
	if err!= nil {
		log.Fatal(err)
	}
	http.HandleFunc("/detect", func(w http.ResponseWriter, r *http.Request) {
		data, err := ioutil.ReadAll(r.Body)
		if err!= nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		threats := cybersecurity.DetectThreats(data)
		fmt.Fprint(w, strings.Join(threats, "\n"))
	})
	log.Fatal(http.ListenAndServe(":8080", nil))
}
