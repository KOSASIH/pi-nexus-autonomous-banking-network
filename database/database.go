package database

import (
	"database/sql"
	"fmt"
	"log"
	"time"

	_ "github.com/lib/pq"
)

type DB struct {
	*sql.DB
}

func NewDB(config *Config) (*DB, error) {
	connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
		config.Host, config.Port, config.User, config.Password, config.DBName)
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, err
	}
	db.SetMaxIdleConns(config.MaxIdleConns)
	db.SetMaxOpenConns(config.MaxOpenConns)
	db.SetConnMaxLifetime(time.Duration(config.ConnMaxLifetime) * time.Second)
	return &DB{db}, nil
}

func (db *DB) Close() error {
	return db.DB.Close()
}

type Config struct {
	Host     string
	Port     int
	User     string
	Password string
	DBName   string
	MaxIdleConns int
	MaxOpenConns int
	ConnMaxLifetime int
}
