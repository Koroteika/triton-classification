# Triton Image Classification Stack

Полнофункциональный проект разворачивания модели классификации изображений через NVIDIA Triton Inference Server. Решение включает FastAPI Gateway, Swagger UI, Streamlit-интерфейс, Prometheus и Grafana-дэшборд.

Модель классифицирует изображения птиц по трем классам:

- `Asian Green Bee-Eater`
- `Brown-Headed Barbet`
- `Common Kingfisher`

## Преимущества архитектуры

### NVIDIA Triton Inference Server

- Запускает ONNX-модель через backend `onnxruntime`.
- Поддерживает dynamic batching для повышения throughput.
- Экспортирует метрики Prometheus на порту `8002`.
- Позволяет хранить модель в стандартной структуре `model_repository/<model_name>/<version>/model.onnx`.

### FastAPI Gateway

- Принимает изображения через `multipart/form-data` или JSON base64.
- Выполняет препроцессинг: RGB, resize `128x128`, нормализация `0..1`.
- Вызывает Triton по gRPC.
- Возвращает предсказанный класс, confidence, probabilities и время инференса.
- Предоставляет Swagger UI для ручной проверки API.

### Streamlit UI

- Позволяет загрузить изображение через браузер.
- Отображает предсказанный класс, уверенность, время Triton, полное время запроса и вероятности по классам.
- Работает как отдельный Docker-сервис.

### Prometheus + Grafana

- Prometheus собирает метрики Triton.
- Grafana автоматически получает datasource и dashboard через provisioning.
- Дэшборд показывает RPS, latency, queue time, pending requests и total requests.

## Стек

| Компонент | Назначение |
|---|---|
| NVIDIA Triton | Инференс ONNX-модели |
| FastAPI | REST API Gateway |
| Streamlit | Пользовательский веб-интерфейс |
| Prometheus | Сбор метрик |
| Grafana | Визуализация метрик |
| Docker Compose | Запуск всего стека |

## Структура проекта

```text
triton-classification/
├── docker-compose.yml
├── README.md
├── test_client.py
├── benchmark_dynamic_batching.py
├── .gitignore
│
├── model_repository/
│   └── image_classifier/
│       ├── config.pbtxt
│       ├── class_names.json
│       └── 1/
│           └── model.onnx
│
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py
│
├── streamlit_app/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
│
├── prometheus/
│   └── prometheus.yml
│
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── datasources.yml
│       └── dashboards/
│           ├── dashboards.yml
│           └── triton.json
│
├── results/
│   ├── dynamic_batching_results.md
│   ├── dynamic_batching_results.json
│   └── dynamic_batching_comparison.svg
│
└── screenshots/
    └── .gitkeep
```

## Быстрый старт

```bash
git clone <ссылка-на-ваш-репозиторий>
cd triton-classification
docker compose up -d --build
```

Если используется старая команда Compose:

```bash
docker-compose up -d --build
```

## Проверка запуска

```bash
docker compose ps
curl http://localhost:8080/health
curl http://localhost:8000/v2/models/image_classifier
```

Ожидаемый ответ `/health`:

```json
{
  "status": "healthy",
  "triton_live": true,
  "model_ready": true,
  "model_name": "image_classifier",
  "input_name": "input_image",
  "output_name": "Identity:0",
  "classes_loaded": 3
}
```

Ожидаемые метаданные модели:

```json
{
  "name": "image_classifier",
  "versions": ["1"],
  "platform": "onnxruntime_onnx",
  "inputs": [{"name": "input_image", "datatype": "FP32", "shape": [-1, 128, 128, 3]}],
  "outputs": [{"name": "Identity:0", "datatype": "FP32", "shape": [-1, 3]}]
}
```

## Сервисы

| Сервис | URL | Описание |
|---|---|---|
| FastAPI | http://localhost:8080 | REST API |
| Swagger UI | http://localhost:8080/docs | Документация и ручной тест `/predict` |
| Streamlit | http://localhost:8501 | Веб-интерфейс классификации |
| Triton HTTP | http://localhost:8000 | Triton HTTP API |
| Triton gRPC | localhost:8001 | Triton gRPC API |
| Triton Metrics | http://localhost:8002/metrics | Метрики для Prometheus |
| Prometheus | http://localhost:9090 | Web UI Prometheus |
| Grafana | http://localhost:3000 | Дэшборды, `admin/admin` |

## API Endpoints

| Метод | Endpoint | Описание |
|---|---|---|
| `GET` | `/` | Информация об API |
| `GET` | `/health` | Проверка состояния Triton и модели |
| `GET` | `/classes` | Список классов |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/predict` | Классификация изображения |

### Пример запроса через файл

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@path/to/image.jpg"
```

### Пример JSON base64

```json
{
  "image": "<base64-encoded-image>"
}
```

### Пример ответа

```json
{
  "model": "image_classifier",
  "batch_size": 1,
  "inference_ms": 181.331,
  "predictions": [
    {
      "class_id": 2,
      "class_name": "Common Kingfisher",
      "confidence": 1.0,
      "probabilities": {
        "Asian Green Bee-Eater": 0.0,
        "Brown-Headed Barbet": 0.0,
        "Common Kingfisher": 1.0
      }
    }
  ]
}
```

## Конфигурация Triton

Файл: `model_repository/image_classifier/config.pbtxt`

```text
name: "image_classifier"
backend: "onnxruntime"
max_batch_size: 32

input [
  {
    name: "input_image"
    data_type: TYPE_FP32
    dims: [128, 128, 3]
  }
]

output [
  {
    name: "Identity:0"
    data_type: TYPE_FP32
    dims: [3]
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 100000
}

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
```

## Streamlit

Запуск только Streamlit-сервиса:

```bash
docker compose up -d --build streamlit
```

Открыть интерфейс:

```text
http://localhost:8501
```

Интерфейс позволяет загрузить изображение птицы и выполнить классификацию через FastAPI/Triton.

## Мониторинг

Prometheus собирает метрики Triton с адреса:

```text
triton:8002/metrics
```

Grafana автоматически загружает:

- datasource: `grafana/provisioning/datasources/datasources.yml`
- dashboard: `grafana/provisioning/dashboards/triton.json`

Основные панели:

| Панель | Метрика |
|---|---|
| Requests per Second | `rate(nv_inference_request_success{model="image_classifier"}[$__rate_interval])` |
| Average Request Duration | `rate(nv_inference_request_duration_us{model="image_classifier"}[$__rate_interval]) / rate(nv_inference_request_success{model="image_classifier"}[$__rate_interval])` |
| Average Queue Time | `rate(nv_inference_queue_duration_us{model="image_classifier"}[$__rate_interval]) / rate(nv_inference_request_success{model="image_classifier"}[$__rate_interval])` |
| Pending Requests | `nv_inference_pending_request_count{model="image_classifier"}` |
| Total Requests | `nv_inference_request_success` и `nv_inference_request_failure` |

## Нагрузочное тестирование

Скрипт:

```bash
python3 -u benchmark_dynamic_batching.py \
  --image "path/to/image.jpg" \
  --requests 1000 \
  --concurrency 50 \
  --warmup 20 \
  --mode all \
  --restore medium \
  --results-dir results
```

Результаты:

| Конфигурация | preferred_batch_size | max_queue_delay_microseconds | Avg Latency (ms) | P95 (ms) | P99 (ms) | Throughput (RPS) | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| No batching | - | - | 743.19 | 895.79 | 925.35 | 65.43 | 1000/1000 |
| Small batch | [2, 4] | 50000 | 387.30 | 494.82 | 538.59 | 126.94 | 1000/1000 |
| Medium batch | [4, 8, 16] | 100000 | 378.23 | 453.48 | 502.01 | 129.62 | 1000/1000 |
| Large batch | [8, 16, 32] | 200000 | 377.22 | 479.11 | 525.50 | 130.34 | 1000/1000 |

Вывод: dynamic batching почти в 2 раза увеличил throughput и снизил среднюю задержку. Наилучший баланс между throughput и latency показала конфигурация `[4, 8, 16]` с `max_queue_delay_microseconds=100000`.

## Скриншоты

Рекомендуемые файлы для отчета и GitHub:

```text
screenshots/swagger-predict.png
screenshots/grafana-dashboard.png
screenshots/streamlit-result.png
```

## Поддержка GPU

В текущей версии проект настроен на CPU:

```text
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
```

Для запуска на NVIDIA GPU нужно установить NVIDIA Container Toolkit и заменить `instance_group` на `KIND_GPU`, а также добавить GPU-reservation в `docker-compose.yml`.

## Требования

- Docker Desktop / Docker Engine
- Docker Compose
- 8 GB RAM желательно для комфортного запуска Triton, Grafana, Prometheus, FastAPI и Streamlit
- macOS/Linux/Windows с Docker

## Остановка

```bash
docker compose stop
```

Полное удаление контейнеров:

```bash
docker compose down
```
