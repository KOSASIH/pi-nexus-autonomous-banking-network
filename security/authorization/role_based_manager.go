package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/casbin/casbin/v2"
	"github.com/casbin/casbin/v2/model"
	"github.com/casbin/casbin/v2/util"
	"github.com/gorilla/mux"
)

type Role struct {
	Name     string   `json:"name"`
	Actions  []string `json:"actions"`
	Children []string `json:"children"`
}

type RoleRequest struct {
	Role *Role `json:"role"`
}

type RoleResponse struct {
	Role *Role `json:"role"`
}

func init() {
	// Load the ABAC model and policy
	model := loadModel("abac_model.conf")
	enforcer, err := casbin.NewEnforcer(model, "abac_policy.csv")
	if err != nil {
		log.Fatalf("Failed to load enforcer: %v", err)
	}

	// Set the enforcer as a global variable
	roleBasedEnforcer = enforcer
}

func main() {
	router := mux.NewRouter()
	router.HandleFunc("/roles/{role}", createRoleHandler).Methods("POST")
	router.HandleFunc("/roles/{role}", getRoleHandler).Methods("GET")
	router.HandleFunc("/roles/{role}", updateRoleHandler).Methods("PUT")
	router.HandleFunc("/roles/{role}", deleteRoleHandler).Methods("DELETE")
	log.Fatal(http.ListenAndServe(":8080", router))
}

func createRoleHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	name := vars["role"]

	var req RoleRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	// Add the role to the policy
	roleBasedEnforcer.AddRoleForUser(name, req.Role.Name)
	for _, action := range req.Role.Actions {
		roleBasedEnforcer.AddPolicy(name, action, name)
	}
	for _, child := range req.Role.Children {
		roleBasedEnforcer.AddRoleForUser(child, name)
	}

	res := &RoleResponse{
		Role: req.Role,
	}
	json.NewEncoder(w).Encode(res)
}

func getRoleHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	name := vars["role"]

	// Get the role from the policy
	role := roleBasedEnforcer.GetRolesForUser(name)
	actions := roleBasedEnforcer.GetFilteredPolicy(0, name, "")
	children := roleBasedEnforcer.GetRolesForUser(name)

	res := &RoleResponse{
		Role: &Role{
			Name:     name,
			Actions:  actions,
			Children: children,
		},
	}
	json.NewEncoder(w).Encode(res)
}

func updateRoleHandler(w http.ResponseWriter, r *http.Request) {
	// Update the role in the policy
	// ...

	res := &RoleResponse{
		Role: &Role{},
	}
	json.NewEncoder(w).Encode(res)
}

func deleteRoleHandler(w http.ResponseWriter, r *http.Request) {
	// Delete the role from the policy
	// ...

	res := &RoleResponse{
		Role: &Role{},
	}
	json.NewEncoder(w).Encode(res)
}

func loadModel(filePath string) model.Model {
	model, err := model.NewModelFromFile(filePath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	return model
}
