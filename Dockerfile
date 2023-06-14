FROM python:3.11.0

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/api_model

COPY main.py .

EXPOSE 8080

CMD ["python", "main.py"]
