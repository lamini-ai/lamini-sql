FROM python:3.11 as base

ARG PACKAGE_NAME="sql-autocomplete"

# Install Ubuntu libraries
RUN apt-get -yq update

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt
RUN pip install -r requirements.txt

# Copy all files to the container
COPY ./scripts /app/${PACKAGE_NAME}/scripts

COPY ./sql-autocomplete.py /app/${PACKAGE_NAME}/sql-autocomplete.py

WORKDIR /app/${PACKAGE_NAME}

RUN chmod a+x /app/${PACKAGE_NAME}/scripts/start.sh

ENV PACKAGE_NAME=$PACKAGE_NAME
ENTRYPOINT ["/app/sql-autocomplete/scripts/start.sh"]

