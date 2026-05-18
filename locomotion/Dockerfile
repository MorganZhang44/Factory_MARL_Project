FROM condaforge/miniforge3:latest

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace/Factory_MARL_Project

COPY locomotion/environment.yml /tmp/locomotion-environment.yml
RUN conda env create -f /tmp/locomotion-environment.yml && conda clean -afy

COPY . /workspace/Factory_MARL_Project

ENV LOCOMOTION_CONDA_ENV=locomotion
ENV PYTHONUNBUFFERED=1

CMD ["bash", "-lc", "./scripts/launch_locomotion.sh"]
