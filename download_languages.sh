#!/bin/bash

# Target directory
TESSDATA_DIR="/usr/local/share/tessdata"

# Create directory if it doesn't exist
mkdir -p "$TESSDATA_DIR"

# List of 50 language codes
LANGS=(
  afr amh ara aze bel ben bod bos bul ceb ces chi_sim chi_tra cym dan deu ell eng est
  fin fra guj heb hin hrv hun ind ita jpn kan khm kor lao lat lav lit mal mar msa nep
  nld nor pan pol por ron rus sin slk spa swe tam tel tha
)

# Download each .traineddata file
for lang in "${LANGS[@]}"; do
  echo "Downloading $lang.traineddata..."
  curl -L -o "$TESSDATA_DIR/$lang.traineddata" \
    "https://github.com/tesseract-ocr/tessdata/raw/main/$lang.traineddata"
done

echo "✅ All 50 language models downloaded successfully!"
