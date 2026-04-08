FROM python:3.10-slim

WORKDIR /code

COPY requirement.txt /code/requirement.txt

# Cai dat cac thu vien can thiet
RUN pip install --no-cache-dir --upgrade -r /code/requirement.txt

# Copy cac thu muc vao container
COPY ./app /code/app
COPY ./model /code/model

# open port 8000
EXPOSE 8000

# Chay FastAPI
# app.main:app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]