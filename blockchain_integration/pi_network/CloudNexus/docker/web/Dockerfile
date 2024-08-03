# Use an official Nginx image as a base
FROM nginx:latest

# Copy the configuration file
COPY nginx.conf /etc/nginx/nginx.conf

# Copy the web application code
COPY . /usr/share/nginx/html

# Expose the port for the web interface
EXPOSE 80
