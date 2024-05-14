package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var vulnerabilities = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Name: "vulnerabilities_total",
		Help: "Total number of vulnerabilities found",
	},
	[]string{"severity"},
)

func init() {
	prometheus.MustRegister(vulnerabilities)
}

func main() {
	http.Handle("/metrics", promhttp.Handler())
	http.HandleFunc("/scan", scanHandler)
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func scanHandler(w http.ResponseWriter, r *http.Request) {
	url := r.URL.Query().Get("url")
	if url == "" {
		http.Error(w, "URL parameter is required", http.StatusBadRequest)
		return
	}

	// Run OWASP ZAP and Selenium to scan the URL
	cmd := exec.Command("python3", "scan.py", url)
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Failed to scan URL: %v\n%s", err, output)
		http.Error(w, "Failed to scan URL", http.StatusInternalServerError)
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

	fmt.Fprintf(w, "Scan completed")
}
