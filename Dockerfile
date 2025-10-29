FROM python:3.14.0-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD [ "fastapi", "dev", "app/main.py", "--host", "0.0.0.0" ]