#!/bin/bash

export PROJECT_ID=$(gcloud config get-value core/project)
export IMAGE_NAME='pytorch_1_10_gpu_eyemind'
export IMAGE_TAG='latest'
export IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}

docker build -f Dockerfile -t ${IMAGE_URI} .

docker push ${IMAGE_URI}