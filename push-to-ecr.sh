#!/bin/bash
set -e

# Use current time as tag
TIMESTAMP=$(date +%Y%m%d%H%M%S)
AWS_REGION="us-west-2"
AWS_ACCOUNT_ID="262976740991"
ECR_REPOSITORY="strands-agent"  # ECR repository name to use
IMAGE_TAG="${TIMESTAMP}"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"

echo "===== AWS ECR Login ====="
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "===== Building Docker Image ====="
docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .

echo "===== Tagging for ECR Repository ====="
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ECR_URI}

echo "===== Pushing Image to ECR Repository ====="
docker push ${ECR_URI}

echo "===== Complete ====="
echo "Image has been successfully built and pushed to ECR."
echo "Image URI: ${ECR_URI}"