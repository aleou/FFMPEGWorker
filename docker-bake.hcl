variable "REGISTRY" {
  default = "aleou"
}

variable "IMAGE" {
  default = "ffmpeg-worker"
}

variable "RELEASE_VERSION" {
  default = "0.1.0"
}

group "default" {
  targets = ["base", "final"]
}

target "base" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "base"
  platforms  = ["linux/amd64"]
  tags       = ["${REGISTRY}/${IMAGE}:${RELEASE_VERSION}-base"]
}

target "final" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "final"
  platforms  = ["linux/amd64"]
  tags       = [
    "${REGISTRY}/${IMAGE}:${RELEASE_VERSION}",
    "${REGISTRY}/${IMAGE}:latest",
  ]
  inherits = ["base"]
}
