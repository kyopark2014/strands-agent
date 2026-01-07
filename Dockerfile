FROM --platform=linux/amd64 python:3.13-slim

WORKDIR /app

RUN pip install streamlit==1.41.0 streamlit-chat pandas numpy boto3 
RUN pip install langchain_aws langchain langchain_community langchain_experimental
RUN pip install mcp wikipedia
RUN pip install strands-agents strands-agents-tools
RUN pip install aioboto3 opensearch-py
RUN pip install tavily-python==0.5.0 yfinance==0.2.52 rizaio==0.8.0 pytz==2024.2 beautifulsoup4==4.12.3
RUN pip install plotly_express==0.4.1 matplotlib==3.10.0 chembl-webresource-client pytrials
RUN pip install PyPDF2==3.0.1 wikipedia requests uv kaleido diagrams reportlab arxiv graphviz sarif-om==1.0.4
RUN pip install rich==13.9.0 bedrock-agentcore

RUN mkdir -p /root/.streamlit
COPY config.toml /root/.streamlit/

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["python", "-m", "streamlit", "run", "application/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
