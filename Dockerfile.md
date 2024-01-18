FROM python:3.11

RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./packages ./packages

RUN poetry install  --no-interaction --no-ansi --no-root

COPY ./app ./app

RUN poetry install --no-interaction --no-ansi

EXPOSE 80

# CMD exec uvicorn app.server:app --host 127.0.0.1 --port 8080

CMD ["uvicorn", "app.server:app", "--host", "127.0.0.1", "--port", "80"]
