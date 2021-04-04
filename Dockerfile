FROM jupyter/datascience-notebook

WORKDIR /home/jovyan/work/

COPY requirements.txt /home/jovyan/work/
RUN pip install -r requirements.txt
RUN R -e "install.packages(c('caTools', 'ggplot2'), repos = 'http://cran.us.r-project.org')"

COPY . /home/jovyan/work

EXPOSE 8888
