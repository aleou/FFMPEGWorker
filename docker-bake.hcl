variable "REGISTRY" {
  default = "aleou"
}

variable "IMAGE" {
  default = "ffmpeg-worker"
}

variable "RELEASE_VERSION" {
  default = "0.7.0"
}

group "default" {
  targets = ["final-api", "final-serverless"]
}

target "base" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "base"
  platforms  = ["linux/amd64"]
  tags       = ["${REGISTRY}/${IMAGE}:${RELEASE_VERSION}-base"]
}

target "final-api" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "final_api"
  platforms  = ["linux/amd64"]
  tags       = [
    "${REGISTRY}/${IMAGE}:${RELEASE_VERSION}-api",
    "${REGISTRY}/${IMAGE}:latest-api",
  ]
  inherits = ["base"]
}

target "final-serverless" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "final_serverless"
  platforms  = ["linux/amd64"]
  tags       = [
    "${REGISTRY}/${IMAGE}:${RELEASE_VERSION}-serverless",
    "${REGISTRY}/${IMAGE}:latest-serverless",
  ]
  inherits = ["base"]
}
