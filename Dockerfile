FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY ./src /app

RUN pip install -r /app/requirements.txt

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]