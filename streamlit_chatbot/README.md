# Claude Sonnet Streamlit 챗봇

AWS Bedrock의 Claude Sonnet 모델을 활용한 Streamlit 기반 챗봇 애플리케이션입니다.

## 기능

- 텍스트 기반 대화
- 이미지 분석 및 설명
- 문서 처리 및 분석
- 스트리밍 응답 지원
- 대화 기록 유지

## 설치 방법

1. 저장소 클론 또는 다운로드

```bash
git clone <repository-url>
cd streamlit_chatbot
```

2. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

3. AWS 자격 증명 설정

`.env.example` 파일을 `.env`로 복사하고 AWS 자격 증명을 입력합니다.

```bash
cp .env.example .env
```

`.env` 파일을 열고 다음 정보를 입력합니다:

```
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1  # 또는 다른 리전
```

## 실행 방법

다음 명령어로 애플리케이션을 실행합니다:

```bash
streamlit run app.py
```

웹 브라우저가 자동으로 열리고 애플리케이션이 표시됩니다.

## 사용 방법

1. 텍스트 메시지 전송
   - 하단의 입력 필드에 메시지를 입력하고 Enter 키를 누릅니다.

2. 이미지 분석
   - "이미지 업로드" 탭에서 이미지를 업로드합니다.
   - 메시지를 입력하고 Enter 키를 누릅니다.
   - Claude Sonnet이 이미지를 분석하고 응답합니다.

3. 문서 처리
   - "문서 업로드" 탭에서 문서를 업로드합니다.
   - 메시지를 입력하고 Enter 키를 누릅니다.
   - Claude Sonnet이 문서를 처리하고 응답합니다.

4. 설정 변경
   - 사이드바에서 스트리밍 응답을 활성화/비활성화할 수 있습니다.
   - "대화 초기화" 버튼을 클릭하여 대화 기록을 지울 수 있습니다.

## 주의 사항

- AWS Bedrock 서비스 사용에는 비용이 발생할 수 있습니다.
- Claude Sonnet 모델에 대한 접근 권한이 필요합니다.
- 대용량 이미지나 문서를 업로드할 경우 처리 시간이 길어질 수 있습니다.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.