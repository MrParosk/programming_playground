FROM ubuntu:18.04

RUN apt-get -y update
RUN apt-get -y install python3.6
RUN apt-get -y install python3-pip
RUN apt-get -y install git
RUN git clone https://github.com/parlstrand/programming_playground
WORKDIR "programming_playground/flask_blog"
RUN pip3 install -r requirements.txt

EXPOSE 5000
ENTRYPOINT [ "python3" ]
CMD [ "run.py" ]