# CTEC
CTEC single-cell RNA-seq ensemble clustering

[![python >3.8.17](https://img.shields.io/badge/python-3.8.17-brightgreen)](https://www.python.org/) 

# Dependences

[![numpy-1.20.3](https://img.shields.io/badge/numpy-1.20.3-red)](https://github.com/numpy/numpy)
[![pandas-1.3.5](https://img.shields.io/badge/pandas-1.3.5-lightgrey)](https://github.com/pandas-dev/pandas)
[![scanpy-1.9.5](https://img.shields.io/badge/scanpy-1.9.5-blue)](https://github.com/theislab/scanpy)

# Datasets
The five benchmark datasets can be download from:

https://drive.google.com/drive/folders/1ZhybTUaBCvIyVY1jiIcIDqTH7SbwXy4Y?usp=drive_link

# Usage
```
ctec_two_method_unknown_cluster.py
```
and
```
ctec_two_method_known_cluster.py
```
are used for two methods ensemble on the benchmark datasets, the Leiden and DESC methods are used by default, with and without knowing the true cluster of datasets.


```
ctec_multiple_method_ensemble.py
```
is used for multiple methods ensemble on the benchmark datasets, and 
```
ctec_multiple_method_ensemble_draw_umap.py
```
is for umap drawing.

# Build and use CTEC docker image
The Docker image for the CTEC method can be obtained through different options to ensure accessibility and reproducibility.
## Option 1
You can run the command
```
docker pull lwchn/ctec:v1
```
on your local machine to obtain the image directly from the Docker Hub repository at https://hub.docker.com/r/lwchn/ctec/tags

## Option 2
Alternatively, you can download the tar file ("ctec_v1.tar") of the Docker image from Google Drive. The file is available at https://drive.google.com/file/d/1GhPMNpS2YpmrFHCPlm-jcOA5FO4v0RUb/view?usp=drive_link

## Option 3
If you prefer, you can create your own CTEC Docker image by using the provided files on the GitHub repository of CTEC. The "Dockerfile" and "requirements.txt" files can be found at the GitHub repository https://github.com/LWCHN/CTEC, with:
```
docker build ./ -t ctec:v1
```

## How to use CTEC docker image
```
docker run -u root -v LOCAL_PATH_OF_GIT_REPO:/ctec_work/:rw ctec:v1 bash -c "python /ctec_work/ctec_two_method_unknown_cluster.py"
docker run -u root -v LOCAL_PATH_OF_GIT_REPO:/ctec_work/:rw ctec:v1 bash -c "python /ctec_work/ctec_two_method_known_cluster.py"
docker run -u root -v LOCAL_PATH_OF_GIT_REPO:/ctec_work/:rw ctec:v1 bash -c "python /ctec_work/ctec_multiple_method_ensemble.py"
docker run -u root -v LOCAL_PATH_OF_GIT_REPO:/ctec_work/:rw ctec:v1 bash -c "python /ctec_work/ctec_multiple_method_ensemble_draw_umap.py"
```
The "LOCAL_PATH_OF_GIT_REPO" should be replaced by local CTEC project path.

# Disclaimer

This tool is for research purposes and not approved for clinical use.

This is not an official Tencent product.

# Coypright

This tool is developed in Tencent AI Lab.

The copyright holder for this project is Tencent AI Lab.

All rights reserved.
