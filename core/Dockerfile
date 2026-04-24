FROM condaforge/miniforge3:latest

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace

COPY core/environment.yml /tmp/core-environment.yml
RUN conda env create -f /tmp/core-environment.yml && conda clean -afy

ENV CORE_CONDA_ENV=core
ENV PYTHONUNBUFFERED=1

CMD ["bash", "-lc", "./scripts/launch_core_dashboard.sh"]
