###############################################################################
# Permissions for SageMaker Training and Automatic Model Tuning (AMT) jobs
#
# Extends the existing SageMaker execution role (referenced, not managed) so
# training containers can write logs and read the DagsHub secret (the secret
# read and S3 access are granted in sagemaker_domain.tf / s3.tf). Job
# submission permissions belong to the GitHub Actions role in github_oidc.tf.
###############################################################################

data "aws_iam_policy_document" "sagemaker_training_runtime" {
  statement {
    sid       = "WriteTrainingLogs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["${aws_cloudwatch_log_group.training_jobs.arn}:*"]
  }
}

resource "aws_iam_role_policy" "sagemaker_training_runtime" {
  name   = "fraud-detection-${var.environment}-training-runtime"
  role   = var.sagemaker_execution_role_name
  policy = data.aws_iam_policy_document.sagemaker_training_runtime.json
}
