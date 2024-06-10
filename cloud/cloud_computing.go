package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"
)

func main() {
	// Create an AWS session
	sess, err := session.NewSession(&aws.Config{Region: aws.String("us-west-2")}, nil)
	if err!= nil {
		log.Fatal(err)
	}

	// Create an EC2 client
	ec2Svc := ec2.New(sess)

	// Launch a new instance
	params := &ec2.RunInstancesInput{
		ImageId:      aws.String("ami-abc123"),
		InstanceType: aws.String("t2.micro"),
		MinCount:     aws.Int64(1),
		MaxCount:     aws.Int64(1),
	}
	resp, err := ec2Svc.RunInstances(params)
	if err!= nil {
		log.Fatal(err)
	}

	// Print the instance ID
	fmt.Println("Instance ID:", *resp.Instances[0].InstanceId)

	// Create an HTTP server
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "Hello, Cloud Banking!")
	})
	log.Fatal(http.ListenAndServe(":8080", nil))
}
