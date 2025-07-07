FROM python:3.12-slim

ARG USERNAME=remoteuser
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

RUN apt-get update && apt-get install -y \
    sudo git graphviz build-essential libgl1-mesa-dev libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev libcairo2-dev libsasl2-dev libldap2-dev \
    curl ca-certificates sudo \
    # && curl -kL https://sh.rustup.rs -sSf | sh -s -- -y \
    # && curl -Ls https://astral.sh/uv/install.sh | sh \
    && apt-get autoremove --purge -y \
    && apt-get clean -y

# create group and user only if not exist
RUN groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}


ENV PATH="/home/${USERNAME}/.local/bin:/home/${USERNAME}/.cargo/bin:${PATH}"

RUN pip install --upgrade pip wheel
RUN pip install uv

CMD ["/bin/sh", "-c", ".venv/bin/python --version 2>/dev/null || uv venv; .venv/bin/uv sync"]
