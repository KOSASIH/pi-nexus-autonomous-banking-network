provider "google" {
  project = "pi-network-gcp"
  region  = "us-west2"
}

# Create a GCP service account for the Pi Network
resource "google_service_account" "pi_network" {
  account_id = "pi-network"
}

# Create a GCP IAM role for the Pi Network
resource "google_project_iam_custom_role" "pi_network" {
  role_id     = "piNetworkRole"
  title       = "Pi Network Role"
  description = "Pi Network custom role"

  permissions = [
    "compute.instances.get",
    "compute.instances.list",
    "storage.buckets.get",
    "storage.buckets.list",
    "bigquery.datasets.get",
    "bigquery.datasets.list",
    "cloudfunctions.functions.get",
    "cloudfunctions.functions.list"
  ]
}

# Create a GCP IAM policy binding for the Pi Network
resource "google_project_iam_binding" "pi_network" {
  project = google_service_account.pi_network.project
  role    = google_project_iam_custom_role.pi_network.id

  members = [
    "serviceAccount:${google_service_account.pi_network.email}"
  ]
}

# Create a GCP IAM key for the Pi Network service account
resource "google_service_account_key" "pi_network" {
  service_account_id = google_service_account.pi_network.name
}

# Output the GCP IAM key
output "pi_network_key" {
  value     = google_service_account_key.pi_network.private_key
  sensitive = true
}
