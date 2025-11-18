FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y build-essential libgl1 && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app", "--workers", "2"]