# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

###############################################################################
# Data lake bucket (referenced, not managed)
#
# The bucket is owned by the legacy Terraform project (state key
# dev/terraform.tfstate), so it is only read here. This config adds the
# raw/, processed/ and models/ prefixes and grants the existing SageMaker
# execution role access to them.
###############################################################################

variable "sagemaker_execution_role_name" {
  description = "Name of the existing SageMaker execution role that needs bucket access."
  type        = string
  default     = "fraud-detection-dev-sagemaker-execution"
}

variable "data_bucket_name" {
  description = "Name of the existing data lake bucket owned by the legacy Terraform project."
  type        = string
  default     = "fraud-detection-dev-data-use1"
}

data "aws_s3_bucket" "data" {
  bucket = var.data_bucket_name
}

resource "aws_s3_object" "prefix" {
  for_each = toset(["raw/", "processed/", "models/"])

  bucket  = data.aws_s3_bucket.data.id
  key     = each.value
  content = ""
}

# Processing/training jobs: read raw + processed, write processed + models.
data "aws_iam_policy_document" "sagemaker_data_access" {
  statement {
    sid       = "ListDataBucket"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation", "s3:ListBucketMultipartUploads"]
    resources = [data.aws_s3_bucket.data.arn]
  }

  statement {
    sid       = "ReadRawData"
    actions   = ["s3:GetObject"]
    resources = ["${data.aws_s3_bucket.data.arn}/raw/*"]
  }

  statement {
    sid = "ReadWriteProcessedAndModels"
    # Spark commits output by writing to a temp path then copy + delete (rename).
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
    resources = [
      "${data.aws_s3_bucket.data.arn}/processed/*",
      "${data.aws_s3_bucket.data.arn}/models/*",
    ]
  }
}

###############################################################################
# Write-once protection for run data
#
# Versioning is already enabled on the bucket (legacy project), so an overwrite
# keeps the old bytes recoverable. These statements add the guarantees that
# versioning alone does not give. The bucket had no policy when this was added,
# so this resource does not replace anything; if the legacy project ever adds
# one, merge the statements instead of applying both.
#
# Not enforced at bucket level: Spark's part files under processed/<run_id>/
# {train,val,test}. Spark (S3A) writes through multipart uploads without
# If-None-Match, so a deny there would break the job. Those are protected by
# errorifexists in spark_job.py plus versioning.
###############################################################################

data "aws_iam_policy_document" "data_bucket_protection" {
  statement {
    sid     = "DenyPermanentDeleteOfRunData"
    effect  = "Deny"
    actions = ["s3:DeleteObjectVersion"]
    resources = [
      "${data.aws_s3_bucket.data.arn}/raw/*",
      "${data.aws_s3_bucket.data.arn}/processed/*",
      "${data.aws_s3_bucket.data.arn}/models/*",
    ]

    principals {
      type        = "*"
      identifiers = ["*"]
    }
  }

  # The launchers write the run marker, config, deps zip, raw manifest and the
  # training source bundle with IfNoneMatch="*"; this makes S3 refuse any write
  # to those keys that does not carry the header, so the guard cannot be skipped.
  statement {
    sid     = "RequireWriteOnceForLaunchArtifacts"
    effect  = "Deny"
    actions = ["s3:PutObject"]
    resources = [
      "${data.aws_s3_bucket.data.arn}/processed/*/config/*",
      "${data.aws_s3_bucket.data.arn}/models/code/*",
    ]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "Null"
      variable = "s3:if-none-match"
      values   = ["true"]
    }
  }
}

resource "aws_s3_bucket_policy" "data_protection" {
  bucket = data.aws_s3_bucket.data.id
  policy = data.aws_iam_policy_document.data_bucket_protection.json
}

resource "aws_iam_role_policy" "sagemaker_data_access" {
  name   = "fraud-detection-${var.environment}-data-bucket-access"
  role   = var.sagemaker_execution_role_name
  policy = data.aws_iam_policy_document.sagemaker_data_access.json
}

output "data_bucket_name" {
  description = "Name of the data lake bucket (use as --bucket for scripts/upload_to_s3.py)."
  value       = data.aws_s3_bucket.data.bucket
}

output "data_bucket_arn" {
  description = "ARN of the data lake bucket."
  value       = data.aws_s3_bucket.data.arn
}
