# User Management
================

The User Management system allows administrators to manage user accounts and profiles.

## Features

* User registration and login
* User profile management
* User role management
* User authentication and authorization

## API Endpoints
--------------

### Users

* **GET /users**: Retrieve a list of all users
* **POST /users**: Create a new user
* **GET /users/:id**: Retrieve a specific user by ID
* **PUT /users/:id**: Update a specific user by ID
* **DELETE /users/:id**: Delete a specific user by ID

### User Roles

* **GET /users/:id/roles**: Retrieve a list of all roles for a specific user
* **POST /users/:id/roles
