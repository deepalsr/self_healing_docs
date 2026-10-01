# Dockerfile

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first, separately from app code, so Docker's layer
# cache can skip reinstalling packages when only app code changes --
# this is the single biggest Docker build-speed win for most projects.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Never bake secrets into the image. Real values are injected at runtime
# via environment variables (docker run --env-file, or the host platform's
# secrets manager) -- same principle as .env never being committed to git.
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]