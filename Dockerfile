FROM python:3.9-slim

# Системные зависимости для umap-learn и scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала копируем только зависимости — слой кешируется при изменении кода
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаём нужные директории (данные монтируются снаружи через volume)
RUN mkdir -p data/raw data/processed image reports/figures

# По умолчанию запускаем веб-приложение.
# Для запуска пайплайна: docker compose run pipeline
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]