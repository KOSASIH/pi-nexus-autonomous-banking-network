# Azure provider variables
azure_subscription_id = "your_azure_subscription_id"
azure_client_id      = "your_azure_client_id"
azure_client_secret = "your_azure_client_secret"
azure_tenant_id      = "your_azure_tenant_id"

# Resource group variables
azure_location = "West US 2"

# Virtual network variables
vnet_address_space = ["10.0.0.0/16"]
vnet_subnet_prefix = "10.0.1.0/24"

# Virtual machine variables
vm_size = "Standard_DS2_v2"
vm_admin_username = "pi-network-admin"
vm_admin_password = "P@ssw0rd!"

# Storage account variables
storage_account_name = "pinetworkstorage"
storage_account_tier = "Standard"
storage_account_replication_type = "LRS"

# Cosmos DB variables
cosmosdb_account_name = "pi-network-cosmosdb"
cosmosdb_offer_type = "Standard"

# AKS cluster variables
aks_cluster_name = "pi-network-aks"
aks_dns_prefix = "pi-network-aks"
aks_node_pool_name = "default"
aks_node_pool_node_count = 1
aks_node_pool_vm_size = "Standard_DS2_v2"
