#!/bin/bash

APP_DIR="$(cd "$(dirname "$0")"/../ && pwd)"
DATA_DIR="$(cd "$APP_DIR/../fusang-data" && pwd)"

NAME="sars-cov-2-omicron_genome"
echo "Running demo for $NAME"
echo "  app dir: $APP_DIR"
echo "  data dir: $DATA_DIR"

#----------------------------------------------------------#

echo "Preparing data for $NAME..."

mkdir -pv ${APP_DIR}/data/${NAME}/

for x in {10..50..10}; do

    if [ -e "${APP_DIR}/data/${NAME}/${NAME}.${x}.fasta" ]; then
        echo "File ${APP_DIR}/data/${NAME}/${NAME}.${x}.fasta already exists, skipping..."
    else
        echo "Creating ${APP_DIR}/data/${NAME}/${NAME}.${x}.fasta from ${DATA_DIR}/${NAME}/${NAME}.fasta..."

        N=$(cat ${DATA_DIR}/${NAME}/${NAME}.fasta | grep "^>" -n | head -n $((x+1)) | tail -n1 | cut -d: -f1)
        N_minus_1=$((N-1))

        cat ${DATA_DIR}/${NAME}/${NAME}.fasta \
            | head -n $N_minus_1 \
            > ${APP_DIR}/data/${NAME}/${NAME}.${x}.fasta
    fi

    if [ -e "${APP_DIR}/data/${NAME}/${NAME}.${x}.aligned.fasta" ]; then
        echo "File ${APP_DIR}/data/${NAME}/${NAME}.${x}.aligned.fasta already exists, skipping..."
    else
        echo "Creating ${APP_DIR}/data/${NAME}/${NAME}.${x}.aligned.fasta from ${APP_DIR}/data/${NAME}/${NAME}.${x}.fasta..."

        mafft ${APP_DIR}/data/${NAME}/${NAME}.${x}.fasta \
            > ${APP_DIR}/data/${NAME}/${NAME}.${x}.aligned.fasta
    fi
done

#----------------------------------------------------------#

mkdir -pv ${APP_DIR}/logs/
mkdir -pv ${APP_DIR}/results/${NAME}/

for x in {10..50..10}; do
    DATE=$(date +"%Y-%m-%d_%H-%M-%S")
    if [ ! -e "${APP_DIR}/results/${NAME}/${NAME}.${x}.tree" ]; then
        (
            echo "Start at $(date)"
            time python ${APP_DIR}/fusang.py \
                infer \
                --input ${APP_DIR}/data/${NAME}/${NAME}.${x}.aligned.fasta \
                --output ${APP_DIR}/results/${NAME}/${NAME}.${x}.tree
            echo "End at $(date)"
        ) \
            > ${APP_DIR}/logs/${DATE}_${NAME}.${x}.log \
            2> ${APP_DIR}/logs/${DATE}_${NAME}.${x}.err
    fi
done

#----------------------------------------------------------#

echo "Demo for $NAME completed."
