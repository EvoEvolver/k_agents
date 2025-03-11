FROM python:3.10
WORKDIR /app
COPY . .

RUN pip install git+https://github.com/ShuxiangCao/LeeQ@k_agents
RUN pip install streamlit

EXPOSE 8080

CMD python -m streamlit run application/leeq/leeq_app.py --server.port=8080 --server.address=0.0.0.0
