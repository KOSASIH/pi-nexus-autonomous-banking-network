#!/bin/bash

# Generate a new private key using OpenSSL
openssl genpkey -algorithm RSA -out private.pem -pkeyopt rsa_keygen_bits:2048

# Extract the public key from the private key
openssl rsa -in private.pem -pubout -outform PEM -out public.pem

# Display the public key
echo "Public key:"
cat public.pem

# Display the private key
echo "Private key:"
cat private.pem
