# Hugging Face Space (Docker) for the poker UI. It clones the GitHub repo at build time, so the Space only
# needs this file and README.md. "Factory rebuild" in the Space settings picks up new commits and bots.
# A private repo needs a GITHUB_TOKEN secret in the Space settings (a read-only fine-grained token).
FROM python:3.13
# the full image (not -slim) already has git

RUN pip install --no-cache-dir uv==0.11.32

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /home/user

ARG REPO=github.com/A-Dzulkipli/poker_bots.git
ARG BRANCH=main
RUN --mount=type=secret,id=GITHUB_TOKEN,mode=0444,required=false \
    if [ -s /run/secrets/GITHUB_TOKEN ]; then auth="x-access-token:$(cat /run/secrets/GITHUB_TOKEN)@"; fi; \
    git clone --depth 1 --branch "$BRANCH" "https://${auth}${REPO}" app \
    && git -C app remote set-url origin "https://${REPO}"

WORKDIR /home/user/app
RUN uv sync --frozen --no-default-groups --group ui

EXPOSE 7860
CMD [".venv/bin/python", "-m", "ui", "--host", "0.0.0.0", "--port", "7860", "--no-browser", "--max-tables", "6"]
