FROM amd64/python:3.9.16-slim

WORKDIR .

COPY requirements.txt ./

RUN apt-get update
RUN apt-get install -y libglib2.0-0 libgl1-mesa-glx
# libsm6 libxrender1 libxext6

RUN pip install -U pip \
&& pip install --no-cache-dir -r ./requirements.txt

# copy other file
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]