# Base 'uv' image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Boilerplate essentials
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copying essential components of application to container
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY configs/ configs/
COPY LICENSE LICENSE

# set working directory in our container to root `corrupt-mnist/`
WORKDIR /

# #  install dependencies
# RUN uv sync --locked --no-cache --no-install-project

# # *Note*: run the following instead to reduce dependency install time
# # by mounting your local uv cache to the Docker image
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

# *Note*: data should be dealt with different when building on the cloud
# COPY data/ data/

# # Verify data was pulled successfully
# RUN test -d /data && find /data -type f | head -5 && echo "✓ Data successfully pulled"

# training script as the entrypoint to our docker image
ENTRYPOINT ["sh", "-c", "uv run src/corrupt_mnist/download_data.py && uv run src/corrupt_mnist/train.py"]
