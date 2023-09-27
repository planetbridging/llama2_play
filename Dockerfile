# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the local content into the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install transformers

# Run app.py when the container launches
CMD ["python", "app.py"]
