FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED 1
ENV WORKERS=10

ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy assets
RUN mkdir /src
WORKDIR /src


RUN apt update -y
RUN apt install -y build-essential -y
RUN apt install -y default-libmysqlclient-dev -y

RUN mkdir /src/logs

FROM base AS dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM dependencies AS development
COPY . /src/
RUN chmod +x /src/run_server.sh

CMD ["./run_server.sh"]