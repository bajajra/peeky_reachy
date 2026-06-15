# Cry-classification service for turing. YAMNet is CPU-friendly; GPU optional.
FROM python:3.11-slim

WORKDIR /srv/peeky
ENV PYTHONUNBUFFERED=1 PEEKY_CRY_PREFER_ML=1 PEEKY_EAGER_LOAD=1

# Install the package (provides the classifier) + service deps + YAMNet stack.
COPY . /srv/peeky
RUN pip install --no-cache-dir -e ".[ml]" \
    && pip install --no-cache-dir -r cry_service/requirements.txt \
    && pip install --no-cache-dir "tensorflow>=2.13" "tensorflow-hub>=0.15"

EXPOSE 8080
CMD ["uvicorn", "cry_service.server:app", "--host", "0.0.0.0", "--port", "8080"]
