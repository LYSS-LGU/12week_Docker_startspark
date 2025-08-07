# Dockerfile

# 1. 베이스 이미지
FROM bitnami/spark:latest

# 2. root 사용자로 전환
USER root

# 3. 한글 폰트 및 필수 도구 설치
RUN apt-get update -y && \
    apt-get install -y fonts-nanum* python3-pip

# 4. Matplotlib 폰트 캐시 초기화
RUN rm -rf /root/.cache/matplotlib/*

# 5. 파이썬 라이브러리 설치
RUN pip install pandas matplotlib seaborn