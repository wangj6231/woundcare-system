# Professor-preview image. Model weights and encrypted preview data are mounted
# at runtime and are intentionally not copied into the image.
FROM node:20-bookworm-slim AS frontend
WORKDIR /frontend
COPY wound_nurse_app/package.json wound_nurse_app/package-lock.json ./
RUN npm ci
COPY wound_nurse_app/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend_main.py woundcare_inference.py ./
COPY --from=frontend /frontend/dist ./wound_nurse_app/dist
RUN useradd --create-home --uid 10001 woundcare && mkdir /data && chown -R woundcare:woundcare /app /data
USER woundcare
EXPOSE 8000
CMD ["python", "backend_main.py"]
