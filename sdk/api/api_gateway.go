package api

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/sdk/api/pb"
	"google.golang.org/grpc"
)

type APIGateway struct {
	pb.UnimplementedAPIGatewayServer
}

func (g *APIGateway) HandleRequest(ctx context.Context, req *pb.Request) (*pb.Response, error) {
	// TO DO: implement API request handling logic
	log.Printf("Received API request from %s", req.ClientID)
	return &pb.Response{Result: "success"}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":50054")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterAPIGatewayServer(srv, &APIGateway{})

	log.Println("API gateway listening on port 50054")
	srv.Serve(lis)

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "PiNexus SDK API Gateway")
	})

	log.Println("HTTP server listening on port 8080")
	http.ListenAndServe(":8080", nil)
}
