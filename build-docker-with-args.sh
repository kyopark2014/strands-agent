#!/bin/bash

# Strands Agent Docker Build Script (with ARG credentials)
echo "🚀 Strands Agent Docker Build Script (with ARG credentials)"
echo "=========================================================="

# Check if AWS CLI is configured
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install AWS CLI first."
    exit 1
fi

# Get AWS credentials from local AWS CLI configuration
echo "📋 Getting AWS credentials from local configuration..."

AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)
AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)
AWS_DEFAULT_REGION=$(aws configure get region)
AWS_SESSION_TOKEN=$(aws configure get aws_session_token)

# Check if credentials are available
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "❌ AWS credentials not found. Please configure AWS CLI first:"
    echo "   aws configure"
    exit 1
fi

echo "✅ AWS credentials found"
echo "   Region: ${AWS_DEFAULT_REGION:-us-east-1}"

# Build Docker image with build arguments
echo ""
echo "🔨 Building Docker image with ARG credentials..."
docker build \
    --platform linux/amd64 \
    --build-arg AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    --build-arg AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    --build-arg AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
    --build-arg AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" \
    -t strands-agent:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully with embedded credentials"
    echo ""
    echo "🚀 To run the container:"
    echo "   docker run -d --name strands-agent-container -p 8501:8501 strands-agent:latest"
    echo ""
    echo "⚠️  Note: AWS credentials are embedded in the Docker image"
    echo "   - Do not share this image publicly"
    echo "   - For production, use environment variables or IAM roles"
else
    echo "❌ Docker build failed"
    exit 1
fi 