FROM python:3.9-slim

WORKDIR /app

RUN pip install streamlit==1.24.0

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]