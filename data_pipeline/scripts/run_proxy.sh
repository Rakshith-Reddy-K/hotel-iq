#!/bin/bash

# UPDATE THESE
PROJECT="hotel-iq"
REGION="us-east4"
INSTANCE="hotel-iq"

# Fixed credential path
CREDENTIALS_FILE="$HOME/.gcp/hoteliq-key.json"

# Run proxy
./cloud-sql-proxy ${PROJECT}:${REGION}:${INSTANCE} \
    --credentials-file=${CREDENTIALS_FILE}