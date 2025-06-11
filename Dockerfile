FROM python:3.11.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TF_AUTOTUNE_RAM_BUDGET_IN_MB=1024

CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8080"]