# The ranking console, self-contained: `docker run` and you are looking at the model.
#
# This is NOT the scoring image. Azure ML's deployment builds its own from
# pipelines/score_environment.yml, which carries azureml-inference-server-http and nothing else.
# This one carries the console, the demonstration data and no Azure dependency at all — it never
# calls an endpoint and holds no credential. Two images because they answer to different callers.
#
# Versions match the local environment that produced every number in the write-up, so a reader
# who runs this gets the same ranking to the last digit rather than something close.

FROM python:3.11-slim

# libgomp is LightGBM's OpenMP runtime and is not in the slim base. Without it the import fails
# at container start with a bare "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      pandas==2.3.3 \
      numpy==2.4.6 \
      lightgbm==4.7.0 \
      pyarrow==24.0.0 \
      python-dotenv==1.2.2

WORKDIR /app
COPY src/ /app/src/
COPY README.md /app/README.md

# The served model, at the layout paths.py expects under RENTAL_RANKING_ROOT.
COPY data/train/serving_bundle/ /app/data/train/serving_bundle/

# The demonstration data: the sealed fold already resolved, plus the coverage report. Built by
# `python -m rental_ranking.cloud.demo --bundle`. Shipping this rather than the feature table
# means the image needs neither the processed layer nor a 25-second re-run of the fold assignment
# on every start — connected components over 44,684 rows, for an answer that cannot change.
COPY data/train/demo_bundle/ /app/demo/

ENV PYTHONPATH=/app/src \
    RENTAL_RANKING_ROOT=/app \
    RENTAL_RANKING_DEMO_BUNDLE=/app/demo \
    PYTHONUNBUFFERED=1

# 0.0.0.0 is correct here and only here: the container's network namespace is the boundary, and
# the run command publishes to the host's loopback. The console's own default stays 127.0.0.1.
EXPOSE 8000
CMD ["python", "-m", "rental_ranking.cloud.console", "--local", "--no-open", "--host", "0.0.0.0"]
