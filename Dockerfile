FROM pytorch/pytorch
WORKDIR /usr/src/app
COPY * ./
RUN pip install gpytorch
COPY . .
CMD ["python", "main.py"]
