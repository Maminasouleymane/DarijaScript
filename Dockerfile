FROM python

WORKDIR /python

COPY . /python

RUN pip install ply

CMD ["python" , "parser.py"]