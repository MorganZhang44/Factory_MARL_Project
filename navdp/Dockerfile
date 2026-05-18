FROM condaforge/miniforge3:latest

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace/Factory_MARL_Project

COPY navdp/environment.yml /tmp/navdp-environment.yml
RUN conda env create -f /tmp/navdp-environment.yml && conda clean -afy

COPY . /workspace/Factory_MARL_Project

ENV NAVDP_CONDA_ENV=navdp
ENV PYTHONUNBUFFERED=1
ENV NAVDP_PLANNER=real
ENV NAVDP_DEVICE=cpu

CMD ["bash", "-lc", "./scripts/launch_navdp.sh"]
