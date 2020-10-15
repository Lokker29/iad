FROM jupyter/datascience-notebook

WORKDIR /home/jovyan/work/

COPY requirements.txt /home/jovyan/work/
RUN pip install -r requirements.txt

COPY . /home/jovyan/work

EXPOSE 8888
