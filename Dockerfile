# MODEL AUTOPSY — engine + API + UI in one container.
#
#   docker build -t model-autopsy .
#   docker run -p 8000:8000 model-autopsy      -> http://localhost:8000
#
# No Node stage: web/static/app.js is committed, so the image only needs
# Python. Rebuild the bundle with `npm run build` before building the image
# if you have changed the front end.
FROM python:3.11-slim

WORKDIR /app

# rdkit/xgboost/scikit-learn wheels are large; install them first so the
# layer caches across source edits
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY autopsy/ ./autopsy/
COPY web/static/ ./web/static/

EXPOSE 8000
CMD ["uvicorn", "autopsy.api:app", "--host", "0.0.0.0", "--port", "8000"]
