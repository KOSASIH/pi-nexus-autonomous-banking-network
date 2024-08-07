#!/bin/bash

# Deploy MCIP service
go run chain/mcip/mcip_service.go &

# Deploy DIAM service
go run identity/diam/diam_service.go &

# Deploy SSSC node
go run consensus/sssc/sssc_node.go &

# Deploy API gateway
go run sdk/api/api_gateway.go &

echo "PiNexus network deployed successfully!"
