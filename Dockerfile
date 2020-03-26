FROM python:3.7-slim-stretch

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# RUN pip install -q git+https://github.com/fastai/fastcore --upgrade

# RUN pip install -q git+https://github.com/fastai/fastai2 --upgrade

RUN pip install -r requirements.txt

RUN pip freeze

COPY app app/

RUN python app/server.py

EXPOSE 80

CMD ["python", "app/server.py", "serve"]
