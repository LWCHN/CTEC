FROM python:3.8.17
WORKDIR /ctec_work
COPY requirements.txt /ctec_work/requirements.txt
RUN apt update && apt install -y graphviz
RUN pip install --no-cache-dir -r requirements.txt
