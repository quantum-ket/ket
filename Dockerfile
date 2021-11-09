FROM ubuntu:20.04

EXPOSE 8888

RUN apt update && apt install python3-pip -y
RUN pip3 install ket-lang kbw numpy scipy jupyter
RUN mkdir -p /home/ket

WORKDIR /home/ket

CMD jupyter notebook --port=8888 --ip 0.0.0.0 --no-browser --allow-root