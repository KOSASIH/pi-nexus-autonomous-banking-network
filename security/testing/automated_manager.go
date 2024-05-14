package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type ScanRequest struct {
	URL string `json:"url"`
}

type ScanResponse struct {
	Vulnerabilities struct {
		Low    int `json:"low"`
		Medium int `json:"medium"`
		High   int `json:"high"`
		Critical int `json:"critical"`
	} `json:"vulnerabilities"`
}

func init() {
	prometheus.MustRegister(vulnerabilities)
}

func main() {
	http.Handle("/metrics", promhttp.Handler())
	http.HandleFunc("/scan", scanHandler)
	http.HandleFunc("/scan_status", scanStatusHandler)
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func scanHandler(w http.ResponseWriter, r *http.Request) {
	var req ScanRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	// Start a new scan goroutine
	done := make(chan struct{})
	go func() {
		scan(req.URL)
		close(done)
	}()

	// Wait for the scan to complete or timeout
	select {
	case <-done:
		fmt.Fprintf(w, "Scan started")
	case <-time.After(10 * time.Second):
		http.Error(w, "Scan timed out", http.StatusRequestTimeout)
	}
}

func scanStatusHandler(w http.ResponseWriter, r *http.Request) {
	// Check the status of the latest scan
	// ...

	// Return the scan results
	res := &ScanResponse{
		Vulnerabilities: struct {
			Low    int `json:"low"`
			Medium int `json:"medium"`
			High   int `json:"high"`
			Critical int `json:"critical"`
		}{
			Low:    0,
			Medium: 0,
			High:   0,
			Critical: 0,
		},
	}
	json.NewEncoder(w).Encode(res)
}

func scan(url string) {
	// Run OWASP ZAP and Selenium to scan the URL
	cmd := exec.Command("python3", "scan.py", url)
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Failed to scan URL: %v\n%s", err, output)
		return
	}

	// Parse the scan results and report the vulnerabilities
	severities := []string{"low", "medium", "high", "critical"}
	for _, severity := range severities {
		count := 0
		for _, line := range strings.Split(string(output), "\n") {
			if strings.Contains(line, fmt.Sprintf("Severity: %s", severity)) {
				count++
			}
		}
		if count > 0 {
			vulnerabilities.WithLabelValues(severity).Add(float64(count))
		}
	}
}
