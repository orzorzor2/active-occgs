#!/bin/bash

### Arguments (personalized) ###
u=${1:-hyzhan}
docker_name=${2:-activelang1}
root_dir=$HOME
g=$(id -gn)
DOCKER_IMAGE=${u}/activelang:1.1

### Run Docker ###
docker run --gpus all --ipc=host \
    --user root \
    --name ${docker_name} \
    -e ROOT_DIR=${root_dir} \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "/nfs:/nfs/" \
    -v "/data0/dataset/Replica:/nfs/home/us000245/datasets/Replica" \
    -v "/data0/dataset/replica_v1:/nfs/STG/SemanticDenseMapping/data/replica_v1/" \    
    -v "/home/zht/data0/ActiveGAMER_v2-main:/nfs/home/us000245/projects" \
    -it $DOCKER_IMAGE /bin/bash
