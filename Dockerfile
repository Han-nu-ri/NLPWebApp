FROM python:3.7

COPY . /docker_work_dir

WORKDIR /docker_work_dir

RUN pip3 install -r requirements.txt

CMD python app.py
