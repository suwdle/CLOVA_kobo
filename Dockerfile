# AI 서비스 (Python)
FROM python:3.12.7 AS flask
WORKDIR /app
# Rust 및 Cargo 설치
RUN apt-get update && apt-get install -y \
    curl \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="$HOME/.cargo/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "app.py"]