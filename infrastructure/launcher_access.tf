# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

###############################################################################
# Data and image access for people who run the launchers from a laptop
#
# src/preprocessing/run_preprocessing_job.py lists raw/<data_version>/ to build
# the raw-input manifest and writes the run's config, deps and manifest;
# src/training/run_training_job.py resolves the training image digest. This
# policy covers exactly those calls. SageMaker job-submission permissions are
# not included: the current launcher user (rec-sys) has AdministratorAccess, so
# nothing is attached by default. List non-admin IAM users here to give them
# the data and image access the launchers need.
###############################################################################

variable "launcher_user_names" {
  description = "Existing IAM users that run the launchers and need the data and image access below (empty = attach to nobody)."
  type        = list(string)
  default     = []
}

data "aws_iam_policy_document" "launcher_access" {
  statement {
    sid       = "ListRawAndProcessed"
    actions   = ["s3:ListBucket"]
    resources = [data.aws_s3_bucket.data.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["raw/*", "processed/*", "models/*"]
    }
  }

  statement {
    sid       = "ReadRawData"
    actions   = ["s3:GetObject"]
    resources = ["${data.aws_s3_bucket.data.arn}/raw/*"]
  }

  statement {
    sid       = "StageRunArtifacts"
    actions   = ["s3:PutObject"]
    resources = ["${data.aws_s3_bucket.data.arn}/processed/*", "${data.aws_s3_bucket.data.arn}/models/code/*"]
  }

  statement {
    sid       = "ResolveTrainingImageDigest"
    actions   = ["ecr:BatchGetImage"]
    resources = ["arn:aws:ecr:${var.aws_region}:${var.sagemaker_image_registry_account}:repository/sagemaker-*"]
  }
}

resource "aws_iam_policy" "launcher_access" {
  name   = "fraud-detection-${var.environment}-launcher-access"
  policy = data.aws_iam_policy_document.launcher_access.json
}

resource "aws_iam_user_policy_attachment" "launcher_access" {
  for_each = toset(var.launcher_user_names)

  user       = each.value
  policy_arn = aws_iam_policy.launcher_access.arn
}
