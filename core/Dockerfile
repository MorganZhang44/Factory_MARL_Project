FROM condaforge/miniforge3:latest

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace/Factory_MARL_Project

COPY core/environment.yml /tmp/core-environment.yml
RUN conda env create -f /tmp/core-environment.yml && conda clean -afy

COPY . /workspace/Factory_MARL_Project

ENV CORE_CONDA_ENV=core
ENV PYTHONUNBUFFERED=1

CMD ["bash", "-lc", "./scripts/launch_core_dashboard.sh"]
