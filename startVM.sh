#!/bin/bash
export IMAGE_FAMILY="tf2-latest-gpu"
export ZONE="asia-east1-c"
export INSTANCE_NAME="deeplearning-instance-tf2"
gcloud compute instances start deeplearning-instance-tf2
sleep 20
gcloud compute ssh --zone=$ZONE jupyter@$INSTANCE_NAME -- -L 8080:localhost:8080
