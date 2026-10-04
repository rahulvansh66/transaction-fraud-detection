# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

###############################################################################
# GitHub Actions -> AWS via OIDC (no stored AWS keys)
#
# The role can be assumed only by workflows of this repository. It may submit
# and inspect SageMaker training/tuning jobs, pass the execution role to
# SageMaker, read processed data and write models, and read the DagsHub secret
# (select_winner.py / evaluate.py log to MLflow). Nothing here can change
# infrastructure: Terraform is applied by a human.
###############################################################################

variable "github_repository" {
  description = "GitHub repository (owner/name) allowed to assume the CI role."
  type        = string
  default     = "rahulvansh66/transaction-fraud-detection"
}

variable "github_subject_patterns" {
  description = "OIDC subject patterns allowed to assume the role, relative to the repository."
  type        = list(string)
  default     = ["*"]
}

variable "create_github_oidc_provider" {
  description = "Set false if the account already has the token.actions.githubusercontent.com provider."
  type        = bool
  # This AWS account already has the provider (created outside this config, confirmed by
  # apply failing with EntityAlreadyExists), so it is only looked up here, never created.
  default = false
}

data "aws_caller_identity" "current" {}

resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 1 : 0

  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

data "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 0 : 1
  url   = "https://token.actions.githubusercontent.com"
}

locals {
  github_oidc_provider_arn = (
    var.create_github_oidc_provider
    ? aws_iam_openid_connect_provider.github[0].arn
    : data.aws_iam_openid_connect_provider.github[0].arn
  )
}

data "aws_iam_policy_document" "github_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.github_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = [for p in var.github_subject_patterns : "repo:${var.github_repository}:${p}"]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "fraud-detection-${var.environment}-github-actions"
  assume_role_policy = data.aws_iam_policy_document.github_assume.json

  tags = {
    project     = "transaction-fraud-detection"
    environment = var.environment
  }
}

data "aws_iam_policy_document" "github_actions" {
  statement {
    sid = "SubmitAndInspectTraining"
    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob",
      "sagemaker:CreateHyperParameterTuningJob",
      "sagemaker:DescribeHyperParameterTuningJob",
      "sagemaker:StopHyperParameterTuningJob",
      "sagemaker:ListTrainingJobsForHyperParameterTuningJob",
      "sagemaker:AddTags",
    ]
    resources = [
      "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:training-job/fraud-*",
      "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:hyper-parameter-tuning-job/hpo-*",
    ]
  }

  statement {
    sid       = "PassExecutionRoleToSageMaker"
    actions   = ["iam:PassRole"]
    resources = [data.aws_iam_role.sagemaker_execution.arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  statement {
    sid       = "ListDataBucket"
    actions   = ["s3:ListBucket"]
    resources = [data.aws_s3_bucket.data.arn]
  }

  statement {
    sid     = "ReadProcessedWriteModels"
    actions = ["s3:GetObject", "s3:PutObject"]
    resources = [
      "${data.aws_s3_bucket.data.arn}/processed/*",
      "${data.aws_s3_bucket.data.arn}/models/*",
    ]
  }

  statement {
    sid       = "ReadDagsHubSecret"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.dagshub_mlflow.arn]
  }
}

resource "aws_iam_role_policy" "github_actions" {
  name   = "fraud-detection-${var.environment}-github-actions"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.github_actions.json
}

output "github_actions_role_arn" {
  description = "Role ARN to store in the GitHub repository variable AWS_OIDC_ROLE_ARN."
  value       = aws_iam_role.github_actions.arn
}
