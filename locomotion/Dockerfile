FROM condaforge/miniforge3:latest

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace

COPY locomotion/environment.yml /tmp/locomotion-environment.yml
RUN conda env create -f /tmp/locomotion-environment.yml && conda clean -afy

ENV LOCOMOTION_CONDA_ENV=locomotion
ENV PYTHONUNBUFFERED=1

CMD ["bash", "-lc", "./scripts/launch_locomotion.sh"]
