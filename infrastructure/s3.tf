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
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [data.aws_s3_bucket.data.arn]
  }

  statement {
    sid       = "ReadRawData"
    actions   = ["s3:GetObject"]
    resources = ["${data.aws_s3_bucket.data.arn}/raw/*"]
  }

  statement {
    sid     = "ReadWriteProcessedAndModels"
    actions = ["s3:GetObject", "s3:PutObject"]
    resources = [
      "${data.aws_s3_bucket.data.arn}/processed/*",
      "${data.aws_s3_bucket.data.arn}/models/*",
    ]
  }
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
