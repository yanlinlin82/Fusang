#!/bin/bash

APP_DIR="$(cd "$(dirname "$0")"/../ && pwd)"
DATA_DIR="$(cd "$APP_DIR/../fusang-data" && pwd)"

NAME="sars-cov-2-omicron_genome"
echo "Running demo for $NAME"
echo "  app dir: $APP_DIR"
echo "  data dir: $DATA_DIR"
