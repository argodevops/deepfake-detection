#!/usr/bin/env bash

API_URL="http://localhost/predict"
BASE_DIR="../testdata"

if ! command -v jq >/dev/null 2>&1; then
  echo "❌ jq is required (sudo apt install jq)"
  exit 1
fi

echo "ground_truth,file,result,score,threshold,input_type,prediction_id"

for label in fake real; do
  DIR="$BASE_DIR/$label"

  if [ ! -d "$DIR" ]; then
    echo "⚠️ Skipping missing directory: $DIR" >&2
    continue
  fi

  find "$DIR" -type f | while read -r file; do
    echo "→ [$label] $file" >&2

    response=$(curl -s -X POST "$API_URL" \
      -F "file=@$file")

    if echo "$response" | jq -e . >/dev/null 2>&1; then
      result=$(echo "$response" | jq -r '.result // "ERROR"')
      score=$(echo "$response" | jq -r '.score // "null"')
      threshold=$(echo "$response" | jq -r '.threshold // "null"')
      input_type=$(echo "$response" | jq -r '.input_type // "unknown"')
      pid=$(echo "$response" | jq -r '.prediction_id // "null"')

      echo "$label,$file,$result,$score,$threshold,$input_type,$pid"
    else
      echo "$label,$file,ERROR,INVALID_JSON,,,,"
    fi
  done
done
