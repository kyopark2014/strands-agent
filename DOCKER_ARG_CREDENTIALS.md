# Docker ARG를 사용한 AWS 자격 증명 설정

## 개요

이 방법은 Docker 빌드 시 `--build-arg`를 사용하여 AWS 자격 증명을 이미지에 포함시키는 방법입니다.

## ⚠️ 보안 주의사항

**이 방법은 개발/테스트 환경에서만 사용하세요!**

- AWS 자격 증명이 Docker 이미지에 하드코딩됩니다
- 이미지를 공유하거나 레지스트리에 푸시할 때 자격 증명이 노출될 수 있습니다
- 프로덕션 환경에서는 절대 사용하지 마세요

## Dockerfile 구성

Dockerfile에서 다음과 같이 ARG를 정의하고 자격 증명을 생성합니다:

```dockerfile
# AWS credentials will be passed at build time via ARG
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION
ARG AWS_SESSION_TOKEN

# Create AWS credentials directory and files
RUN mkdir -p /root/.aws

# Create credentials file from build args
RUN if [ ! -z "$AWS_ACCESS_KEY_ID" ] && [ ! -z "$AWS_SECRET_ACCESS_KEY" ]; then \
        echo "[default]" > /root/.aws/credentials && \
        echo "aws_access_key_id = $AWS_ACCESS_KEY_ID" >> /root/.aws/credentials && \
        echo "aws_secret_access_key = $AWS_SECRET_ACCESS_KEY" >> /root/.aws/credentials && \
        if [ ! -z "$AWS_SESSION_TOKEN" ]; then \
            echo "aws_session_token = $AWS_SESSION_TOKEN" >> /root/.aws/credentials; \
        fi && \
        chmod 600 /root/.aws/credentials; \
    fi

# Create config file
RUN echo "[default]" > /root/.aws/config && \
    echo "region = ${AWS_DEFAULT_REGION:-us-east-1}" >> /root/.aws/config && \
    echo "output = json" >> /root/.aws/config && \
    chmod 600 /root/.aws/config
```

## 사용 방법

### 1. AWS CLI 구성 (이미 되어 있다면 생략)
```bash
aws configure
```

### 2. Docker 이미지 빌드
```bash
./build-docker-with-args.sh
```

### 3. Docker 컨테이너 실행
```bash
./run-docker-args.sh
```

## 수동 빌드 방법

스크립트를 사용하지 않고 직접 빌드하려면:

```bash
# AWS 자격 증명 가져오기
AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)
AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)
AWS_DEFAULT_REGION=$(aws configure get region)
AWS_SESSION_TOKEN=$(aws configure get aws_session_token)

# Docker 빌드
docker build \
    --platform linux/amd64 \
    --build-arg AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    --build-arg AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    --build-arg AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
    --build-arg AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" \
    -t strands-agent:latest .

# 컨테이너 실행
docker run -d \
    --platform linux/amd64 \
    --name strands-agent-container \
    -p 8501:8501 \
    strands-agent:latest
```

## 자격 증명 테스트

컨테이너 내부에서 AWS 자격 증명이 올바르게 설정되었는지 확인:

```bash
# 컨테이너 내부에서 AWS CLI 테스트
docker exec -it strands-agent-container aws sts get-caller-identity

# 자격 증명 파일 확인
docker exec -it strands-agent-container cat /root/.aws/credentials
docker exec -it strands-agent-container cat /root/.aws/config
```

## 장점

1. **빌드 시점에 자격 증명 설정**: 런타임에 환경 변수를 전달할 필요 없음
2. **자동화 가능**: 스크립트로 자동화 가능
3. **단순함**: 컨테이너 실행 시 추가 설정 불필요

## 단점

1. **보안 위험**: 이미지에 자격 증명이 하드코딩됨
2. **이미지 공유 불가**: 자격 증명이 포함된 이미지는 공유 불가
3. **자격 증명 갱신 어려움**: 자격 증명이 변경되면 이미지를 다시 빌드해야 함

## 대안 (프로덕션용)

### 1. 환경 변수 사용
```bash
docker run -e AWS_ACCESS_KEY_ID=xxx -e AWS_SECRET_ACCESS_KEY=xxx ...
```

### 2. 볼륨 마운트
```bash
docker run -v ~/.aws:/root/.aws:ro ...
```

### 3. IAM 역할 사용 (EC2, ECS, EKS)
```bash
# 컨테이너가 IAM 역할을 사용하도록 설정
```

### 4. AWS Secrets Manager
```bash
# 런타임에 자격 증명을 가져오도록 애플리케이션 수정
```

## 파일 구조

```
strands-agent/
├── Dockerfile                    # ARG 방식으로 수정됨
├── build-docker-with-args.sh     # 빌드 스크립트
├── run-docker-args.sh           # 실행 스크립트
└── DOCKER_ARG_CREDENTIALS.md    # 이 문서
```

## 문제 해결

### 빌드 실패 시:
1. AWS CLI가 올바르게 구성되었는지 확인
2. 자격 증명이 유효한지 확인: `aws sts get-caller-identity`
3. Docker 빌드 로그 확인: `docker build --progress=plain ...`

### 자격 증명이 작동하지 않는 경우:
1. 컨테이너 내부에서 자격 증명 파일 확인
2. AWS CLI 테스트: `docker exec -it container aws sts get-caller-identity`
3. 권한 확인: `docker exec -it container ls -la /root/.aws/` 