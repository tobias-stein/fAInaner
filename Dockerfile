FROM public.ecr.aws/lambda/python:3.8

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY data/train ${LAMBDA_TASK_ROOT}/data/train
COPY env ${LAMBDA_TASK_ROOT}/env
COPY pipeline ${LAMBDA_TASK_ROOT}/pipeline
COPY services ${LAMBDA_TASK_ROOT}/services
COPY config.py ${LAMBDA_TASK_ROOT}
COPY hyper.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "hyper.handler" ]