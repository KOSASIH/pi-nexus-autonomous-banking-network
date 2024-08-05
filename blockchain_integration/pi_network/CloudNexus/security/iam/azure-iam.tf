# Create an Azure key vault for the Pi Network
resource "azurerm_key_vault" "pi_network" {
  name                = "pi-network-kv"
  resource_group_name = "pi-network-rg"
  location            = "West US"
  sku_name           = "standard"

  access_policy {
    tenant_id = azurerm_ad_service_principal.pi_network.tenant_id
    object_id = azurerm_ad_service_principal.pi_network.id

    key_permissions = [
      "Get",
      "List",
      "Create",
      "Delete",
      "Update",
      "Import",
      "Export",
    ]

    secret_permissions = [
      "Get",
      "List",
      "Set",
      "Delete",
    ]
  }
}

# Create an Azure key vault secret for the Pi Network
resource "azurerm_key_vault_secret" "pi_network" {
  name      = "pi-network-secret"
  value     = "your_secret_value"
  vault_uri = azurerm_key_vault.pi_network.vault_uri
}

# Output the Azure key vault secret
output "pi_network_secret" {
  value     = azurerm_key_vault_secret.pi_network.value
  sensitive = true
}
