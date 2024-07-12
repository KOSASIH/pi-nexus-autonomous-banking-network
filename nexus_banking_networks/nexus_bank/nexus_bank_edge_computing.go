package main

import (
	"context"
	"fmt"
	"log"

	edge "github.com/edge-computing/edge-sdk-go"
)

func main() {
	// Create an Edge Computing client
	client, err := edge.NewClient("https://edge-computing.example.com", "my-api-key")
	if err != nil {
		log.Fatal(err)
	}

	// Define a function to process data at the edge
	func processEdgeData(ctx context.Context, data []byte) ([]byte, error) {
		// Process the data using a machine learning model
		model, err := edge.LoadModel("my-model", "latest")
		if err != nil {
			return nil, err
		}
		output, err := model.Predict(data)
		if err != nil {
			return nil, err
		}
		return output, nil
	}

	// Register the function with the Edge Computing client
	err = client.RegisterFunction("process-edge-data", processEdgeData)
	if err != nil {
		log.Fatal(err)
	}

	// Deploy the function to the edge
	err = client.DeployFunction("process-edge-data", "my-device")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Function deployed to the edge!")
}
