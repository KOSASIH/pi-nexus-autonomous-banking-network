# Use an official Python image as the base image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the required packages
RUN pip install -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Set the command to run the application
CMD ["python", "web_app.py"]
