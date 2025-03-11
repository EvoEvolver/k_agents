FROM python:3.10
WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/ShuxiangCao/LeeQ.git

EXPOSE 8080

CMD python -m streamlit run application/leeq/leeq_app.py --server.port=8080 --server.address=0.0.0.0
