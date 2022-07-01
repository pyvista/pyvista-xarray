ARG BASE_IMAGE_TAG=latest
FROM ghcr.io/pyvista/pyvista:$BASE_IMAGE_TAG

USER root
COPY . /opt/pvxarray/
WORKDIR /opt/pvxarray/
RUN pip install -r requirements.txt
RUN pip install .

USER jovyan
WORKDIR ${HOME}
COPY tests/data/bahamas_rgb.tif ${HOME}/bahamas_rgb.tif
COPY examples/ ${HOME}
ENV PYVISTA_OFF_SCREEN=true
