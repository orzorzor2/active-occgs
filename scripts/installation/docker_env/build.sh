#!/bin/bash

echo "[Installation] | start building docker image ..."

### Arguments (personalized) ###
PYTHON_VERSION=3.9
USER_NAME=freezing
DOCKER_TAG=gamer-based-gs

### Docker building ###
echo "Will build docker container $DOCKER_TAG ..."
docker build \
    --file envs/Dockerfile \
    --tag $DOCKER_TAG \
    --force-rm \
    --build-arg USER_NAME=${USER_NAME} \
    --build-arg python=${PYTHON_VERSION} \
    .
