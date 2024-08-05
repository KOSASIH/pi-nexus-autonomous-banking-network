provider "aws" {
  region = "us-west-2"
}

# Create an IAM user for the Pi Network
resource "aws_iam_user" "pi_network" {
  name = "pi-network"
}

# Create an IAM role for the Pi Network
resource "aws_iam_role" "pi_network" {
  name        = "pi-network"
  description = "Pi Network role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Effect = "Allow"
      }
    ]
  })
}

# Create an IAM policy for the Pi Network
resource "aws_iam_policy" "pi_network" {
  name        = "pi-network"
  description = "Pi Network policy"

  policy      = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "ec2:*",
          "s3:*",
          "dynamodb:*",
          "lambda:*",
          "cloudwatch:*"
        ]
        Resource = "*"
        Effect    = "Allow"
      }
    ]
  })
}

# Attach the IAM policy to the IAM role
resource "aws_iam_role_policy_attachment" "pi_network" {
  role       = aws_iam_role.pi_network.name
  policy_arn = aws_iam_policy.pi_network.arn
}

# Create an IAM access key for the Pi Network user
resource "aws_iam_access_key" "pi_network" {
  user    = aws_iam_user.pi_network.name
}

# Output the IAM access key ID and secret
output "pi_network_access_key_id" {
  value = aws_iam_access_key.pi_network.id
}

output "pi_network_access_key_secret" {
  value     = aws_iam_access_key.pi_network.secret
  sensitive = true
}
