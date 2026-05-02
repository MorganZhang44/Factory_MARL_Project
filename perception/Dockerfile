FROM condaforge/miniforge3:latest

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace

COPY perception/environment.yml /tmp/perception-environment.yml
RUN conda env create -f /tmp/perception-environment.yml && conda clean -afy

ENV PERCEPTION_CONDA_ENV=perception
ENV PYTHONUNBUFFERED=1

CMD ["bash", "-lc", "./scripts/launch_perception.sh"]
