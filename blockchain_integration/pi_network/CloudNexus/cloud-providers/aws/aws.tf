# Configure the AWS provider
provider "aws" {
  region = var.aws_region
  access_key = var.aws_access_key
  secret_key = var.aws_secret_key
}

# Create a VPC
resource "aws_vpc" "pi_network" {
  cidr_block = "10.0.0.0/16"
  tags = {
    Name = "Pi Network VPC"
  }
}

# Create a subnet
resource "aws_subnet" "pi_network" {
  cidr_block = "10.0.1.0/24"
  vpc_id     = aws_vpc.pi_network.id
  availability_zone = "us-west-2a"
  tags = {
    Name = "Pi Network Subnet"
  }
}

# Create a security group
resource "aws_security_group" "pi_network" {
  name        = "pi-network-sg"
  description = "Security group for Pi Network"
  vpc_id      = aws_vpc.pi_network.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Create an EC2 instance
resource "aws_instance" "pi_network" {
  ami           = "ami-0c94855ba95c71c99"
  instance_type = "t2.micro"
  vpc_security_group_ids = [aws_security_group.pi_network.id]
  subnet_id = aws_subnet.pi_network.id
  key_name               = "pi-network-key"

  tags = {
    Name = "Pi Network EC2 Instance"
  }
}

# Create an RDS instance
resource "aws_db_instance" "pi_network" {
  identifier = "pi-network-rds"
  engine     = "mysql"
  instance_class = "db.t2.micro"
  allocated_storage = 20
  db_name  = "pi_network_db"
  username = "pi_network_user"
  password = "pi_network_password"
  vpc_security_group_ids = [aws_security_group.pi_network.id]
  db_subnet_group_name = aws_db_subnet_group.pi_network.name
}

# Create an S3 bucket
resource "aws_s3_bucket" "pi_network" {
  bucket = "pi-network-bucket"
  acl    = "private"

  tags = {
    Name        = "Pi Network S3 Bucket"
    Environment = "Dev"
  }
}

# Create a CloudWatch log group
resource "aws_cloudwatch_log_group" "pi_network" {
  name = "pi-network-log-group"
}

# Create a CloudWatch log stream
resource "aws_cloudwatch_log_stream" "pi_network" {
  name           = "pi-network-log-stream"
  log_group_name = aws_cloudwatch_log_group.pi_network.name
}
