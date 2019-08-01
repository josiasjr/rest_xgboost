FROM debian:stable-slim

WORKDIR /app

COPY bin /app/crd-frd
COPY app.yaml /app

#COPY libgomp1_6.3.0-18+deb9u1_amd64.deb /app
#RUN dpkg -i libgomp1_6.3.0-18+deb9u1_amd64.deb && rm libgomp1_6.3.0-18+deb9u1_amd64.deb

#EXPOSE 8080

CMD bin/start.sh
