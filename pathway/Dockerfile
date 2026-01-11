FROM python:3.10-slim

WORKDIR /app

# Install Pathway
RUN pip install --upgrade pip
RUN pip install pathway[all]

# Copy code
COPY . .

CMD ["python", "pathway_pipeline.py"]
