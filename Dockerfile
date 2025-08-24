FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libxml2-dev \
        libxslt1-dev \
        libblas-dev \
        liblapack-dev \
        gfortran \
        build-essential \
        python3-dev \
        python3-pip \
        libffi-dev \
        libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir obspy numpy scipy pytz pandas tqdm pyarrow

WORKDIR /app
COPY app/* .
ENTRYPOINT ["python", "detect_segments.py"]

