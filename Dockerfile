# Install pytorch
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-10

WORKDIR /app

COPY ./setup.cfg .
COPY ./pyproject.toml .
COPY ./eyemind ./eyemind
COPY ../OBF ./OBF

RUN pip install .

COPY ./notebooks ./notebooks

