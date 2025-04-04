# Sentiment Analysis API

FastAPI-приложение для анализа тональности текста, использующее модель `blanchefort/rubert-base-cased-sentiment` из Hugging Face Transformers.

---

## Быстрый старт (через Docker Compose)

### Требования

- Docker
- Docker Compose

### Запуск

1. Клонируй репозиторий:

```bash
git clone https://github.com/your-username/sentiment-api.git
cd sentiment-api
```

2.Построй и запусти контейнер:

```bash
docker-compose up --build
```

3.API будет доступно по адресу:

```
http://localhost:8000
```

---

## Примеры запросов

### GET `/`

```bash
curl http://localhost:8000/
```

**Ответ:**

```json
{
  "message": "Hello World! Use POST /analyze to analyze text sentiment."
}
```

---

### POST `/analyze`

```bash
curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"content": "This was awesome!"}'
```

**Ответ:**

```json
{
  "label": "POSITIVE",
  "score": 0.9987
}
```

---
