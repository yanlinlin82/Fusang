#!/bin/bash

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

# fetch latest code
git pull

# run demo
bash demo-1.sh
bash demo-2.sh
bash demo-3.sh
