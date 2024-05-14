package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var accessControlRequests = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Name: "access_control_requests_total",
		Help: "Total number of access control requests",
	},
	[]string{"result"},
)

func init() {
	prometheus.MustRegister(accessControlRequests)
}

func main() {
	http.Handle("/metrics", promhttp.Handler())
	http.HandleFunc("/access_control", accessControlHandler)
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func accessControlHandler(w http.ResponseWriter, r *http.Request) {
	result := r.URL.Query().Get("result")
	if result == "" {
		http.Error(w, "Result parameter is required", http.StatusBadRequest)
		return
	}

	accessControlRequests.WithLabelValues(result).Inc()

	fmt.Fprintf(w, "Access control result: %s", result)
}
