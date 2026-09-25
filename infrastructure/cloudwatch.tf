###############################################################################
# CloudWatch logging and container access for SageMaker Processing jobs
#
# Owns the processing-job log group (explicit retention instead of "Never
# expire") and grants the existing SageMaker execution role the permissions
# a Spark Processing job needs to write logs and pull the SageMaker Spark
# image from ECR.
###############################################################################

variable "processing_log_retention_days" {
  description = "Retention in days for the SageMaker Processing job log group."
  type        = number
  default     = 14
}

resource "aws_cloudwatch_log_group" "processing_jobs" {
  name              = "/aws/sagemaker/ProcessingJobs"
  retention_in_days = var.processing_log_retention_days

  tags = {
    project     = "transaction-fraud-detection"
    environment = var.environment
  }
}

data "aws_iam_policy_document" "sagemaker_processing_runtime" {
  statement {
    sid       = "WriteProcessingLogs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["${aws_cloudwatch_log_group.processing_jobs.arn}:*"]
  }

  statement {
    sid       = "EcrAuthToken"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "PullSageMakerImages"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "sagemaker_processing_runtime" {
  name   = "fraud-detection-${var.environment}-processing-runtime"
  role   = var.sagemaker_execution_role_name
  policy = data.aws_iam_policy_document.sagemaker_processing_runtime.json
}
