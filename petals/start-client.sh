#!/bin/sh
sudo docker container stop public-client
sudo docker container remove public-client
sudo docker create -p 8080:8080 --ipc host --gpus 1 --volume petals-cache3:/root/.cache --name public-client hive-chat:mistral
sudo docker container start public-client