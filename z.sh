#!/usr/bin/env bash
if [ "$1" == "" ]; then
  echo "Please specify a python script file. Usage: z.sh <script_file_location>"
  exit 1
fi

if [ ! -f "$1" ]; then
  echo "File does not exist: $1"
  exit 1
fi

poetry lock && poetry install --no-root
TF_ENABLE_ONEDNN_OPTS=0 # Option to disable one DNN library which prevents GPU usage \
poetry run python3 "$1"

exit 0
