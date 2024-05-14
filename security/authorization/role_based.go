package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/casbin/casbin/v2"
	"github.com/casbin/casbin/v2/model"
	"github.com/casbin/casbin/v2/util"
)

func main() {
	// Load the ABAC model and policy
	model := loadModel("abac_model.conf")
	enforcer, err := casbin.NewEnforcer(model, "abac_policy.csv")
	if err != nil {
		log.Fatalf("Failed to load enforcer: %v", err)
	}

	// Define user roles and permissions
	roles := map[string][]string{
		"admin": {"read", "write", "delete"},
		"user":  {"read", "write"},
	}

	// Define a resource and an action
	resource := "document"
	action := "read"

	// Check if the user has the required permission
	user := "alice"
	role := "user"
	permissions := roles[role]
	hasPermission := util.Contains(permissions, action)

	// Enforce the access control and authorization
	sub := fmt.Sprintf("%s:%s", user, role)
	obj := resource
	act := action
	result, err := enforcer.Enforce(sub, obj, act)
	if err != nil {
		log.Fatalf("Failed to enforce access control: %v", err)
	}

	// Print the result
	if result {
		fmt.Println("Access granted")
	} else {
		fmt.Println("Access denied")
	}
}

func loadModel(filePath string) model.Model {
	model, err := model.NewModelFromFile(filePath)
if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	return model
}
