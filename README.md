# CTEC
CTEC single-cell RNA-seq ensemble clustering

[![python >3.8.17](https://img.shields.io/badge/python-3.8.17-brightgreen)](https://www.python.org/) 

# Dependences

[![numpy-1.20.3](https://img.shields.io/badge/numpy-1.20.3-red)](https://github.com/numpy/numpy)
[![pandas-1.3.5](https://img.shields.io/badge/pandas-1.3.5-lightgrey)](https://github.com/pandas-dev/pandas)
[![scanpy-1.9.5](https://img.shields.io/badge/scanpy-1.9.5-blue)](https://github.com/theislab/scanpy)

# Datasets
https://drive.google.com/drive/folders/1ZhybTUaBCvIyVY1jiIcIDqTH7SbwXy4Y?usp=drive_link

# Usage
run ctec_two_method_ensemble.py for two methods ensemble on the benchmark datasets, the Leiden and DESC methods are used by default.

run ctec_multiple_method_ensemble for multiple methods ensemble on the benchmark datasets.

# Build docker image
1. How to download/pull CTEC docker image
The docker image can be downloaded from https://hub.docker.com/r/lwchn/ctec/tags by 
docker pull lwchn/ctec:v1

2. How to build CTEC docker image
Alternately, this image can be build by Dockerfile with
docker build ./ -t ctec:v1

3. How to use CTEC docker image
docker run -u root -v LOCAL_PATH_OF_GIT_REPO:/ctec_work/:rw ctec:v1 bash -c "python /ctec_work/ctec_two_method_unknown_cluster.py"

docker run -u root -v LOCAL_PATH_OF_GIT_REPO:/ctec_work/:rw ctec:v1 bash -c "python /ctec_work/ctec_two_method_known_cluster.py"

# Disclaimer

This tool is for research purpose and not approved for clinical use.

This is not an official Tencent product.

# Coypright

This tool is developed in Tencent AI Lab.

The copyright holder for this project is Tencent AI Lab.

All rights reserved.
