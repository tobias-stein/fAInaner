#!/usr/bin/env bash

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1
AWS_ECR=${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com

IMG_NAME=hyper
IMG_TAG=latest

IMG_LOCAL=${IMG_NAME}:${IMG_TAG}
IMG_REMOT=${AWS_ECR}/${IMG_NAME}:${IMG_TAG}

# authenticate
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ECR} > /dev/null 2>&1

# make sure remote repository exists
aws ecr describe-repositories --repository-names ${IMG_NAME} > /dev/null 2>&1
if ! [ $? -eq 0 ];
then
    echo "Remote repository not found. Creating new repository '${IMG_NAME}' ..."
    aws ecr create-repository               \
        --repository-name ${IMG_NAME}       \
        --image-scanning-configuration scanOnPush=true \
        --image-tag-mutability MUTABLE      \
        --no-cli-pager
fi

# tag new image and push to remote registry
docker tag ${IMG_LOCAL} ${IMG_REMOT}
docker push ${IMG_REMOT}

aws lambda get-function --function-name ${IMG_NAME} > /dev/null 2>&1
if ! [ $? -eq 0 ];
then
    LAMBDA_EX_ROLE=${IMG_NAME}-lambda-ex

    aws iam create-role                     \
        --role-name ${LAMBDA_EX_ROLE}       \
        --assume-role-policy-document '{"Version": "2012-10-17","Statement": [{ "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}' \
        --no-cli-pager

    aws lambda create-function              \
        --region ${AWS_REGION}              \
        --function-name ${IMG_NAME}         \
        --architectures arm64               \
        --package-type Image                \
        --code ImageUri=${IMG_REMOT}        \
        --timeout 900                       \
        --role arn:aws:iam::${AWS_ACCOUNT}:role/${LAMBDA_EX_ROLE}  \
        --no-cli-pager
else
    aws lambda update-function-code         \
        --region ${AWS_REGION}              \
        --function-name ${IMG_NAME}         \
        --image-uri ${IMG_REMOT}            \
        --architectures arm64               \
        --no-cli-pager
fi