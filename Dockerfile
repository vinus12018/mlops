FROM python:3.12-slim

WORKDIR /code

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY ./app /code/app

# MLflow 및 DagsHub 연동을 위한 환경 변수 자리표시자
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_TRACKING_USERNAME=""
ENV MLFLOW_TRACKING_PASSWORD=""

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]