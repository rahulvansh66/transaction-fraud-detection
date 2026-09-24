###############################################################################
# Terraform and AWS provider configuration
#
# Region comes from the same value as config/env/<env>.yaml (aws.region).
###############################################################################

terraform {
  required_version = ">= 1.5"

  # Shared state bucket/lock table live in ap-south-1; the key is separate from
  # the legacy project's dev/terraform.tfstate.
  backend "s3" {
    bucket         = "fraud-detection-dev-tfstate"
    key            = "dev/ml-foundation/terraform.tfstate"
    region         = "ap-south-1"
    dynamodb_table = "fraud-detection-dev-tflock"
    encrypt        = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

provider "aws" {
  region = var.aws_region
}
