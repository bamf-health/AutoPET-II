FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel



RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown -R algorithm:algorithm /opt/algorithm /input /output 

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"
ENV nnUNet_raw_data_base="nnUNet_raw_data_base/"
ENV RESULTS_FOLDER="nnUNet_trained_models/"
ENV nnUNet_preprocessed="nnUNet_preprocessed/"
RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt
# COPY --chown=algorithm:algorithm nnUNet_raw_data_base /opt/algorithm/nnUNet_raw_data_base/
RUN mkdir -p /opt/algorithm/nnUNet_raw_data_base
COPY --chown=algorithm:algorithm nnUNet_trained_models /opt/algorithm/nnUNet_trained_models/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@