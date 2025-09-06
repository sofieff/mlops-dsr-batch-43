# set python version for the base image
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /code

# Copy the dependencies file to the working directory
COPY requirements.txt /code/requirements.txt

# Install any dependencies
RUN pip install -r /code/requirements.txt

# Copy the content of the local src directory (the entire project) to the working directory
COPY ./app /code/app

ENV WANDB_API_KEY=""
ENV WANDB_ORG=""
ENV WANDB_PROJECT=""
ENV WANDB_MODEL_NAME=""
ENV WANDB_MODEL_VERSION=""

# Expose the port FastAPI is running on, port 8080
EXPOSE 8080 

# this is the command that will be run when the container starts
CMD ["fastapi", "run", "app/main.py", "--port", "8080", "--reload"]
