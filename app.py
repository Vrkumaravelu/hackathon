from flask import Flask, request, jsonify
import requests
import filetype
import io
import os
import json
from openai import OpenAI
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import pdfplumber
from docx import Document
from tqdm import tqdm 
import tempfile
import gc

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key="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")

# --- AI-powered language detection, translation, and NER ---
def ai_translate_with_openai(raw_text, target_language='English'):
    try:
        prompt = f"""
You are a multilingual assistant.

1. Detect the language of the following text and provide:
   - languageCode (ISO 639-1)
   - languageName
   - confidence (0.0 to 1.0)

2. Translate it to '{target_language}'.

3. Identify named entities (people, places, organizations, dates).

Text:
\"\"\"{raw_text}\"\"\"

Respond in the following JSON format:
{{
  "languageCode": "...",
  "languageName": "...",
  "confidence": 0.0,
  "translatedText": "...",
  "accuracyScore": 0.0,
  "entities": ["..."]
}}
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a translation, language detection, and entity extraction expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )

        content = response.choices[0].message.content.strip()

        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```", 1)[1].split("```", 1)[0].strip()

        return json.loads(content)

    except Exception as e:
        print("OpenAI error:", str(e))
        return {
            "languageCode": "unknown",
            "languageName": "unknown",
            "confidence": 0.0,
            "translatedText": "[Translation failed]",
            "accuracyScore": 0.0,
            "entities": [],
            "error": str(e)
        }

# --- File readers ---
def extract_text_from_pdf(buffer):
    text = ""

    try:
        with pdfplumber.open(io.BytesIO(buffer)) as pdf:
            print(f"PDF contains {len(pdf.pages)} pages. Extracting text...")
            for i, page in enumerate(tqdm(pdf.pages, desc="📝 Text Extraction Progress")):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        print("Error using pdfplumber:", e)

    if not text.strip():
        print("No text found using pdfplumber. Falling back to OCR...")

        # Convert PDF to images
        images = convert_from_bytes(buffer)
        print(f"Total OCR pages: {len(images)}")

        tessdata_dir = "/usr/local/share/tessdata"
        lang_files = [f.replace(".traineddata", "") for f in os.listdir(tessdata_dir) if f.endswith(".traineddata")]
        lang_string = "+".join(lang_files) if lang_files else "eng"

        for idx, image in enumerate(tqdm(images, desc="OCR Progress")):
            try:
                ocr_text = pytesseract.image_to_string(image, lang=lang_string, config=f'--tessdata-dir {tessdata_dir}')
                text += ocr_text + "\n"
            except Exception as ocr_err:
                print(f"OCR failed on page {idx + 1}: {ocr_err}")
            finally:
                del image
                gc.collect()

    return text


def extract_text_from_docx(buffer):
    doc = Document(io.BytesIO(buffer))
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_image(buffer):
    # image = Image.open(io.BytesIO(buffer))
    # return pytesseract.image_to_string(image, lang='tha+eng')
    image = Image.open(io.BytesIO(buffer))

    # Try detecting all languages installed in tessdata directory
    tessdata_dir = "/usr/local/share/tessdata"
    lang_files = [f.replace(".traineddata", "") for f in os.listdir(tessdata_dir) if f.endswith(".traineddata")]

    lang_string = "+".join(lang_files) if lang_files else "eng"

    print(f"[INFO] Using languages for OCR: {lang_string}")

    return pytesseract.image_to_string(image, lang=lang_string, config=f'--tessdata-dir {tessdata_dir}')

def extract_text(buffer, mime):
    if "pdf" in mime:
        return extract_text_from_pdf(buffer)
    elif "word" in mime or "officedocument" in mime:
        return extract_text_from_docx(buffer)
    elif mime.startswith("image/"):
        return extract_text_from_image(buffer)
    elif mime.startswith("text/"):
        return buffer.decode("utf-8")
    else:
        raise Exception("Unsupported file type")

# --- Main endpoint ---
@app.route('/process', methods=['POST'])
def process_document():
    data = request.get_json()
    url = data.get('url')
    target_language = data.get('target_language', 'English')

    if not url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': '*/*',
        'Referer': 'https://taza-hackathon-s3.s3.ap-southeast-1.amazonaws.com/'
    }

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=10, verify=False)
        if resp.status_code != 200:
            return jsonify({
                "error": f"Failed to fetch content from URL.",
                "status_code": resp.status_code,
                "reason": resp.reason
            }), 400
        buffer = resp.content
    except Exception as e:
        return jsonify({"error": f"Download failed: {str(e)}"}), 400

    kind = filetype.guess(buffer)
    mime = kind.mime if kind else resp.headers.get('content-type', 'application/octet-stream')
    file_name = url.split("/")[-1] or "document"

    try:
        text = extract_text(buffer, mime)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if not text.strip():
        return jsonify({"error": "Could not extract any text from the document."}), 400

    ai_result = ai_translate_with_openai(text, target_language)

    return jsonify({
        "original_text": text,
        "detected_language_code": ai_result.get("languageCode"),
        "detected_language_name": ai_result.get("languageName"),
        "confidence": ai_result.get("confidence"),
        "translation": ai_result.get("translatedText"),
        "accuracy_score": ai_result.get("accuracyScore"),
        "entities": ai_result.get("entities"),
        "file_name": file_name,
        "file_type": mime
    })

if __name__ == '__main__':
    app.run(debug=True)
