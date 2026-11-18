FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JOB_SCOUT_HOME=/data

RUN groupadd --gid 1000 jobscout \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash jobscout \
    && mkdir -p /data \
    && chown -R jobscout:jobscout /data

WORKDIR /app

COPY pyproject.toml README.md LICENSE job_scout.py ./
COPY jobscout ./jobscout

RUN pip install --no-cache-dir ".[pdf,xml]"

VOLUME ["/data"]
USER jobscout

ENTRYPOINT ["job-scout"]
CMD ["status"]
