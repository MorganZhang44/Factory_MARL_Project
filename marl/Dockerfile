FROM condaforge/miniforge3:latest

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace

COPY marl/environment.yml /tmp/marl-environment.yml
RUN conda env create -f /tmp/marl-environment.yml && conda clean -afy

ENV MARL_CONDA_ENV=marl
ENV PYTHONUNBUFFERED=1

CMD ["bash", "-lc", "./scripts/launch_marl.sh"]
