###############################################################################
# DagsHub MLflow credentials
#
# Secrets Manager entry holding the DagsHub tracking credentials that
# SageMaker jobs read at runtime and export as MLFLOW_TRACKING_USERNAME /
# MLFLOW_TRACKING_PASSWORD. The tracking URI is not secret and lives in
# config/env/{dev,prod}.yaml.
###############################################################################

variable "environment" {
  description = "Deployment environment name (dev or prod)."
  type        = string
}

variable "dagshub_username" {
  description = "DagsHub username used as MLFLOW_TRACKING_USERNAME."
  type        = string
}

variable "dagshub_token" {
  description = "DagsHub access token used as MLFLOW_TRACKING_PASSWORD. Supply via TF_VAR_dagshub_token, never commit."
  type        = string
  sensitive   = true
}

variable "secret_recovery_window_days" {
  description = "Days Secrets Manager retains a deleted secret (0 = immediate delete, for dev only)."
  type        = number
  default     = 7
}

resource "aws_secretsmanager_secret" "dagshub_mlflow" {
  name                    = "fraud-detection/${var.environment}/dagshub-mlflow"
  description             = "DagsHub MLflow tracking credentials for SageMaker jobs"
  recovery_window_in_days = var.secret_recovery_window_days

  tags = {
    project     = "transaction-fraud-detection"
    environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "dagshub_mlflow" {
  secret_id = aws_secretsmanager_secret.dagshub_mlflow.id
  secret_string = jsonencode({
    MLFLOW_TRACKING_USERNAME = var.dagshub_username
    MLFLOW_TRACKING_PASSWORD = var.dagshub_token
  })
}

# Attach to the SageMaker execution role once that role is defined in Terraform.
data "aws_iam_policy_document" "read_dagshub_secret" {
  statement {
    sid       = "ReadDagsHubMlflowSecret"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.dagshub_mlflow.arn]
  }
}

output "dagshub_secret_arn" {
  description = "ARN of the DagsHub MLflow credentials secret."
  value       = aws_secretsmanager_secret.dagshub_mlflow.arn
}
