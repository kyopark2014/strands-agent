import os
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import streamlit as st

def load_env_variables():
    """
    .env 파일에서 환경 변수를 로드합니다.
    """
    load_dotenv()

def get_bedrock_client():
    """
    AWS Bedrock 클라이언트를 생성하고 반환합니다.
    """
    try:
        return boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
    except ClientError as e:
        st.error(f"AWS 클라이언트 생성 중 오류 발생: {str(e)}")
        return None

def check_aws_credentials():
    """
    AWS 자격 증명이 설정되어 있는지 확인합니다.
    """
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not access_key or not secret_key:
        st.warning("AWS 자격 증명이 설정되지 않았습니다. .env 파일을 확인하세요.")
        return False
    return True