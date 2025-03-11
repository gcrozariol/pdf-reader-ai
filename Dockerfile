# Use the AWS base image for Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Install build-essential compiler and tools
RUN microdnf update -y && microdnf install -y gcc-c++ make

# Copy the requirements.txt file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy PDF file
COPY limiters.pdf ${LAMBDA_TASK_ROOT}/limiters.pdf

# Copy the Lambda function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Set permissions for the Lambda function code
RUN chmod +x app.py

# Set CMD to your handler
CMD ["app.handler"]