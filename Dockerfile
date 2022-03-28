FROM public.ecr.aws/lambda/python:3.8

RUN yum install -y git htop atop mc
RUN yum -y install gcc-c++

COPY . ${LAMBDA_TASK_ROOT}

RUN pip3 install -r requirements.txt

CMD ["main.lambda_handler"]
