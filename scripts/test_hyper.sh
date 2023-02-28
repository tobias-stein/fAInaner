IMG_NAME=hyper
IMG_TAG=latest
PORT=9000

# kill old running container instance, if any
ContainerID=$(docker ps -f name=${IMG_NAME} --format "{{.ID}"})
if [[ $ContainerID ]] 
then
    docker kill $ContainerID  
    docker rm $ContainerID  
fi

# spin up a new container instance
docker run --name ${IMG_NAME} -d -p ${PORT}:8080 ${IMG_NAME}:${IMG_TAG}

# send some test data
RESPONSE=$(curl -XPOST "http://localhost:${PORT}/2015-03-31/functions/function/invocations" -d '{"trial_id": 123, "model_name": "a2c", "hyper_params": {"model_args": {"ent_coef": 0.0, "learning_rate": 5e-06}, "policy_args": {"net_arch": [64, 64], "optimizer_class": "AdamW"}}}')


echo "Response was ${RESPONSE}"

# kill container instance
ContainerID=$(docker ps -f name=${IMG_NAME} --format "{{.ID}"})
docker kill $ContainerID  
docker rm $ContainerID