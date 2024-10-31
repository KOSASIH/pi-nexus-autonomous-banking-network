# API Documentation

## Base URL

[http://localhost:3000/api](http://localhost:3000/api) 

## Authentication
### Register User
- **Endpoint**: `/users/register`
- **Method**: `POST`
- **Request Body**:
    ```json
    1 {
    2     "username": "string",
    3     "password": "string"
    4 }
    ```
- **Response**:
    - **201 Created**: User registered successfully.
    - **400 Bad Request**: User already exists.

### Login User
- **Endpoint**: `/users/login`
- **Method**: `POST`
- **Request Body**:
    ```json
    1 {
    2     "username": "string",
    3     "password": "string"
    4 }
    ```
- **Response**:
    - **200 OK**: Returns a JWT token.
    - **401 Unauthorized**: Invalid credentials.

### Get User Profile
- **Endpoint**: `/users/profile`
- **Method**: `GET`
- **Headers**:
    - `Authorization: Bearer <token>`
- **Response**:
    - **200 OK**: Returns user profile data.

## Contract Endpoints
### Get Contract Details
- **Endpoint**: `/contracts/:contractAddress`
- **Method**: `GET`
- **Response**:
    - **200 OK**: Returns contract details.
    - **500 Internal Server Error**: Error fetching contract details.

### Interact with Contract
- **Endpoint**: `/contracts/:contractAddress/interact`
- **Method**: `POST`
- **Request Body**:
    ```json
    1 {
    2     "value": "uint256"
    3 }
    ```
- **Response**:
    - **200 OK**: Interaction successful, returns transaction hash.
    - **500 Internal Server Error**: Error interacting with contract.
