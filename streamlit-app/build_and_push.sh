#!/bin/bash

# ECR 저장소 URI
ECR_REPOSITORY_URI="262976740991.dkr.ecr.us-west-2.amazonaws.com/streamlit-app"
REGION="us-west-2"

# ECR 로그인
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPOSITORY_URI

# Docker 이미지 빌드 (x86_64 플랫폼 지정)
docker build --platform linux/amd64 -t streamlit-app .

# 이미지에 태그 지정
docker tag streamlit-app:latest $ECR_REPOSITORY_URI:latest

# ECR에 이미지 푸시
docker push $ECR_REPOSITORY_URI:latest

echo "Image successfully pushed to $ECR_REPOSITORY_URI:latest"