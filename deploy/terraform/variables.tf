variable "region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "eu-central-1"
}

variable "project" {
  description = "Name prefix and tag applied to all resources."
  type        = string
  default     = "nnforhjb"
}

variable "instance_type" {
  description = "EC2 instance type for the MLflow server. t3.small is ample for a single sequential logger."
  type        = string
  default     = "t3.small"
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB. Holds the SQLite backend store, so size for run-metadata growth."
  type        = number
  default     = 10
}

variable "mlflow_port" {
  description = "Port the MLflow server listens on (bound to localhost; reached via SSM port-forward)."
  type        = number
  default     = 5000
}

variable "python_minor" {
  description = "Python minor version used for the MLflow server venv. Keep aligned with pyproject.toml requires-python."
  type        = string
  default     = "3.12"
}

variable "mlflow_package_spec" {
  description = "MLflow pip requirement for the server. Keep aligned with pyproject.toml dependencies."
  type        = string
  default     = "mlflow>=2.20"
}

variable "mlflow_version" {
  description = "Optional exact MLflow version override. Empty string uses mlflow_package_spec."
  type        = string
  default     = ""
}

variable "subnet_id" {
  description = "Subnet to launch into. Defaults to the first subnet of the account's default VPC."
  type        = string
  default     = null
}
