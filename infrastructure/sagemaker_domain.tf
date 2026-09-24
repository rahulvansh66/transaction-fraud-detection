###############################################################################
# SageMaker Studio domain
#
# Single-user IAM-auth domain in the default VPC, reusing the existing
# SageMaker execution role. Studio traffic goes over the public internet
# (PublicInternetOnly) so notebooks and jobs can reach DagsHub MLflow without
# a NAT gateway. Also grants that role read access to the DagsHub secret.
###############################################################################

variable "studio_user_profile_name" {
  description = "Name of the Studio user profile created in the domain."
  type        = string
  default     = "rahul"
}

data "aws_iam_role" "sagemaker_execution" {
  name = var.sagemaker_execution_role_name
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_sagemaker_domain" "studio" {
  domain_name             = "fraud-detection-${var.environment}"
  auth_mode               = "IAM"
  vpc_id                  = data.aws_vpc.default.id
  subnet_ids              = data.aws_subnets.default.ids
  app_network_access_type = "PublicInternetOnly"

  default_user_settings {
    execution_role = data.aws_iam_role.sagemaker_execution.arn
  }

  tags = {
    project     = "transaction-fraud-detection"
    environment = var.environment
  }
}

resource "aws_sagemaker_user_profile" "default" {
  domain_id         = aws_sagemaker_domain.studio.id
  user_profile_name = var.studio_user_profile_name
}

resource "aws_iam_role_policy" "read_dagshub_secret" {
  name   = "fraud-detection-${var.environment}-read-dagshub-secret"
  role   = var.sagemaker_execution_role_name
  policy = data.aws_iam_policy_document.read_dagshub_secret.json
}

output "studio_domain_id" {
  description = "ID of the SageMaker Studio domain."
  value       = aws_sagemaker_domain.studio.id
}
