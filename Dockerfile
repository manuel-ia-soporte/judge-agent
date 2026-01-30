FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./pyproject.toml
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        fastapi==0.115.6 \
        uvicorn==0.30.6 \
        pydantic==2.12.5 \
        model-library==0.1.7 \
        aiohttp==3.11.18 \
        backoff==2.2.1 \
        beautifulsoup4==4.12.3 \
        python-dotenv==1.2.1

COPY src ./src
COPY cache ./cache
COPY data ./data
COPY scripts ./scripts

ENV PYTHONPATH=/app/src
ENV FINANCE_GREEN_URL=http://localhost:9009

EXPOSE 9009

ENTRYPOINT ["python", "-m", "finance_green_agent.server"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
