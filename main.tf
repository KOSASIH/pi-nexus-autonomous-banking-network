provider "aws" {
  region = "us-west-2"
}

locals {
  common_tags = {
    Terraform = "true"
    Environment = "production"
  }
}

module "autonomous_banking_network" {
  source = "KOSASIH/pi-nexus-autonomous-banking-network/aws"
  version = "0.1.0"

  name = "my-autonomous-banking-network"
  region = "us-west-2"
  vpc_cidr = "10.0.0.0/16"

  subnets = {
    private = {
      cidr_blocks = ["10.0.1.0/24", "10.0.2.0/24"]
      availability_zones = ["us-west-2a", "us-west-2b"]
    }
    public = {
      cidr_blocks = ["10.0.101.0/24", "10.0.102.0/24"]
      availability_zones = ["us-west-2a", "us-west-2b"]
    }
  }

  security_groups = {
    ssh = {
      ingress = [
        {
          from_port = 22
          to_port = 22
          protocol = "tcp"
          cidr_blocks = ["0.0.0.0/0"]
        }
      ]
    }
    http = {
      ingress = [
        {
          from_port = 80
          to_port = 80
          protocol = "tcp"
          cidr_blocks = ["0.0.0.0/0"]
        }
      ]
    }
    https = {
      ingress = [
        {
          from_port = 443
          to_port = 443
          protocol = "tcp"
          cidr_blocks = ["0.0.0.0/0"]
        }
      ]
    }
  }

  additional_components = {
    bastion_host = {
      ami = "ami-0c94855ba95c574c8" # Amazon Linux 2 AMI (HVM), SSD Volume Type
      instance_type = "t3.micro"
      key_name = "my-key-pair"
      subnet_id = module.autonomous_banking_network.public_subnets[0]
      security_groups = [module.autonomous_banking_network.security_group_ssh[0]]
    }
    database = {
      engine = "postgres"
      engine_version = "13.4"
      instance_class = "db.t3.micro"
      name = "my-database"
      username = "my-username"
      password = "my-password"
      vpc_security_group_ids = [module.autonomous_banking_network.security_group_https[0]]
      subnet_ids = module.autonomous_banking_network.private_subnets
      skip_final_snapshot = true
    }
    load_balancer = {
      name = "my-load-balancer"
      internal = false
      security_groups = [module.autonomous_banking_network.security_group_https[0]]
      subnets = module.autonomous_banking_network.public_subnets
      enable_deletion_protection = true
    }
  }

  tags = merge(local.common_tags, {
    Name = "my-autonomous-banking-network"
  })
}

output "vpc_id" {
  value = module.autonomous_banking_network.vpc_id
}

output "private_subnets" {
  value = module.autonomous_banking_network.private_subnets
}

output "public_subnets" {
  value = module.autonomous_banking
