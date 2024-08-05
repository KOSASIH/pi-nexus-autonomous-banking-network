# Configure the Google Cloud provider
provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
  credentials = file(var.gcp_credentials_file)
}

# Create a VPC
resource "google_compute_network" "pi_network" {
  name                    = "pi-network-vpc"
  auto_create_subnetworks = "false"
}

# Create a subnet
resource "google_compute_subnetwork" "pi_network" {
  name          = "pi-network-subnet"
  ip_cidr_range = "10.0.1.0/24"
  network       = google_compute_network.pi_network.self_link
  region        = var.gcp_region
}

# Create a firewall rule
resource "google_compute_firewall" "pi_network" {
  name    = "pi-network-firewall"
  network = google_compute_network.pi_network.self_link

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["0.0.0.0/0"]
}

# Create a Compute Engine instance
resource "google_compute_instance" "pi_network" {
  name         = "pi-network-instance"
  machine_type = "n1-standard-1"
  zone         = var.gcp_zone

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-9"
    }
  }

  network_interface {
    network = google_compute_network.pi_network.self_link
    access_config {
      // Ephemeral IP
    }
  }

  metadata = {
    ssh-keys = "pi-network-user:${file("~/.ssh/pi-network-key.pub")}"
  }
}

# Create a Cloud SQL instance
resource "google_sql_database_instance" "pi_network" {
  name                = "pi-network-sql"
  region              = var.gcp_region
  database_version    = "MYSQL_5_7"
  deletion_protection = false

  settings {
    tier = "db-n1-standard-1"
  }
}

# Create a Cloud Storage bucket
resource "google_storage_bucket" "pi_network" {
  name          = "pi-network-bucket"
  location      = var.gcp_region
  storage_class = "REGIONAL"
}

# Create a Cloud Pub/Sub topic
resource "google_pubsub_topic" "pi_network" {
  name = "pi-network-topic"
}

# Create a Cloud Pub/Sub subscription
resource "google_pubsub_subscription" "pi_network" {
  name  = "pi-network-subscription"
  topic = google_pubsub_topic.pi_network.name
}

# Create a Cloud Logging sink
resource "google_logging_sink" "pi_network" {
  name        = "pi-network-sink"
  destination = "pubsub.googleapis.com/projects/${var.gcp_project}/topics/${google_pubsub_topic.pi_network.name}"
  filter      = "resource.type=\"gce_instance\""
}
