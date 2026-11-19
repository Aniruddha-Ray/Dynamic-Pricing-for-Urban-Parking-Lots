# Linux image with the full stack, including Pathway (which has no Windows wheels).
#
#   docker build -t parking-pricing .
#   docker run --rm -v "$PWD/outputs:/app/outputs" parking-pricing stream        # Pathway pipeline
#   docker run --rm -p 8000:8000 parking-pricing api --host 0.0.0.0              # REST API
#   docker run --rm -p 5006:5006 parking-pricing dashboard --address 0.0.0.0     # live dashboard
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY parking_pricing ./parking_pricing
RUN pip install ".[all,dev]"

COPY dataset.csv ./
COPY tests ./tests

EXPOSE 8000 5006
ENTRYPOINT ["parking-pricing"]
CMD ["--help"]
