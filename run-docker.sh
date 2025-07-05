#!/bin/bash

# Strands Agent Docker Run Script (for ARG-built images)
echo "🚀 Strands Agent Docker Run Script"
echo "=================================="

# Check if image exists
if ! docker images | grep -q "strands-agent.*latest"; then
    echo "❌ Docker image 'strands-agent:latest' not found."
    echo "   Please build the image first using:"
    echo "   ./build-docker-with-args.sh"
    exit 1
fi

# Stop and remove existing container if it exists
echo "🧹 Cleaning up existing container..."
docker stop strands-agent-container 2>/dev/null || true
docker rm strands-agent-container 2>/dev/null || true

# Run Docker container
echo ""
echo "🚀 Starting Docker container..."
docker run -d \
    --platform linux/amd64 \
    --name strands-agent-container \
    -p 8501:8501 \
    strands-agent:latest

if [ $? -eq 0 ]; then
    echo "✅ Container started successfully!"
    echo ""
    echo "🌐 Access your application at: http://localhost:8501"
    echo ""
    echo "📊 Container status:"
    docker ps | grep strands-agent-container
    echo ""
    echo "📝 To view logs: docker logs strands-agent-container"
    echo "🛑 To stop: docker stop strands-agent-container"
    echo "🗑️  To remove: docker rm strands-agent-container"
    echo ""
    echo "🔍 To test AWS credentials in container:"
    echo "   docker exec -it strands-agent-container aws sts get-caller-identity"
else
    echo "❌ Failed to start container"
    exit 1
fi 