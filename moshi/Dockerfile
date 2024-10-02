# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# Assuming you have a requirements.txt file in the moshi directory
RUN pip install --no-cache-dir -r requirements.txt

# Install Moshi and gradio
RUN pip install --no-cache-dir moshi gradio

# Expose the port used by the server
EXPOSE 8998

# Set environment variable for the model (with a default value)
ENV HF_REPO=kyutai/moshiko-pytorch-bf16

# Run the server when the container launches
CMD python -m moshi.server --gradio-tunnel --hf-repo $HF_REPO
