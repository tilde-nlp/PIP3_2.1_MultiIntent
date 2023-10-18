FROM python:3.8-buster

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip \
  && unzip v0.9.2.zip \
  && mv fastText-0.9.2 fastText \
  && sed -i 's=vec.mul(1.0 / norm);=//vec.mul(1.0 / norm); //better vectors by commenting this=g' fastText/src/fasttext.cc

RUN cd fastText && make

COPY . /app

ENV port="12345"
ENV vector_file="/app/data/mono.lv.100.bin"

ENTRYPOINT ["sh", "-c", "python src/VectorizerService.py ./fastText/fasttext ${vector_file} ${port}"]
