package main

import (
	"context"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
)

type ctxKey struct{}

func getField(r *http.Request, index int) string {
	fields := r.Context().Value(ctxKey{}).([]string)
	return fields[index]
}

type apiWidgetPart struct {
	slug string
	id   int
}

func (h apiWidgetPart) update(w http.ResponseWriter, r *http.Request) {
	slug := getField(r, 0)
	id, _ := strconv.Atoi(getField(r, 1))
	fmt.Fprintf(w, "apiUpdateWidgetPart %s %d\n", slug, id)
}

func (h apiWidgetPart) delete(w http.ResponseWriter, r *http.Request) {
	slug := getField(r, 0)
	id, _ := strconv.Atoi(getField(r, 1))
	fmt.Fprintf(w, "apiDeleteWidgetPart %s %d\n", slug, id)
}

func apiUpdateWidgetPartHandler(w http.ResponseWriter, r *http.Request) {
	h := apiWidgetPart{getField(r, 0), getField(r, 1)}
	h.update(w, r)
}

func apiDeleteWidgetPartHandler(w http.ResponseWriter, r *http.Request) {
	h := apiWidgetPart{getField(r, 0), getField(r, 1)}
	h.delete(w, r)
}

func main() {
	r := mux.NewRouter()

	r.HandleFunc("/api/widgets/{slug}/parts/{id:[0-9]+}/update", apiUpdateWidgetPartHandler).Methods("POST")
	r.HandleFunc("/api/widgets/{slug}/parts/{id:[0-9]+}/delete", apiDeleteWidgetPartHandler).Methods("POST")

	http.ListenAndServe(":8080", r)
}
