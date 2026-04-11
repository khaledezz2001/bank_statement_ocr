import runpod
import json
import base64
import re
import io
import os
import torch
from pdf2image import convert_from_bytes
from PIL import Image
from vllm import LLM, SamplingParams

def log(msg):
    print(f"[LOG] {msg}", flush=True)

# ===============================
# CONFIG & MODEL LOADING (vLLM)
# ===============================
MODEL_PATH = "/models/qwen3-vl"
MAX_PAGES_PER_BATCH = 5  # Qwen3-VL supports multi-image natively

log("Loading Qwen3-VL-30B-A3B-Instruct with vLLM (bfloat16)...")

llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",
    max_model_len=16384,
    max_num_seqs=2,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": MAX_PAGES_PER_BATCH},
)

sampling_params = SamplingParams(
    max_tokens=4096,
    temperature=0,
)

log("Qwen3-VL-30B-A3B-Instruct loaded successfully with vLLM.")

# ===============================
# PROMPT
# ===============================
SYSTEM_PROMPT = """You are a helpful financial assistant.
Your task is to extract all transaction details from the provided bank statement images.
Return ONLY a valid JSON array of objects. Do not include any markdown formatting (like ```json).

Output Format Examples:

Example 1 (separate Debits/Credits columns):
[
  {
    "date": "2014-05-15",
    "description": "DIVIDEND",
    "debit": null,
    "credit": 1495.80,
    "balance": 514894.75,
    "currency": "USD"
  }
]

Example 2 (single Amount column with negative values):
[
  {
    "date": "2025-07-01",
    "description": "IBU-Low Activity Fees For June 2025",
    "debit": 23.46,
    "credit": null,
    "balance": 5105.29,
    "currency": "USD"
  }
]

Rules:
1. Extract every single transaction row.
2. If a value is missing, use null.
3. Ensure numbers are floats (no currency symbols or thousand separators). Use absolute values (always positive).
4. Date format: ALWAYS output dates as YYYY-MM-DD. IMPORTANT: Bank statements almost always use DD/MM/YYYY format (day first, then month). For example, 02/06/2025 means June 2nd 2025 (2025-06-02), NOT February 6th. Even if both day and month values are 12 or below, assume DD/MM/YYYY. Use the statement's date range header and the description context (e.g. "Fees For May" posted in June) to confirm.
5. CAREFULLY check the column headers to determine whether an amount is a debit or credit:
   - If there are separate "Debits" and "Credits" columns, look at which column the number appears under.
   - If there is a single "Amount" column, negative values (with a minus sign) are DEBITS and positive values are CREDITS.
   - Fees, charges, and withdrawals are always DEBITS.
6. "description" should contain the transaction type/name and any meaningful details (including any reference codes, voucher numbers, or transaction IDs found in the row).
7. "currency" is the currency of the account as shown on the statement header or transaction details (e.g. USD, EUR, GBP, SAR, AED, CHF). Detect it from the statement context.
8. Output ONLY these 6 fields per transaction: date, description, debit, credit, balance, currency. Do NOT include any other fields.
"""


def repair_truncated_json(text):
    """Attempt to repair truncated JSON arrays by finding the last complete object."""
    start = text.find('[')
    if start == -1:
        return None
    
    last_brace = text.rfind('}')
    if last_brace == -1:
        return None
    
    truncated = text[start:last_brace + 1].rstrip().rstrip(',')
    repaired = truncated + '\n]'
    
    try:
        data = json.loads(repaired)
        if isinstance(data, list):
            log(f"Repaired truncated JSON: recovered {len(data)} transactions.")
            return data
    except json.JSONDecodeError:
        pass
    
    return None


def pil_to_data_uri(image):
    """Convert PIL Image to a base64 data URI for vLLM."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def process_pages(images):
    """
    Process multiple pages in a single prompt using Qwen3-VL's
    native multi-image support via vLLM.
    """
    # Resize images for consistency
    processed_images = []
    for img in images:
        if max(img.size) > 2000:
            img.thumbnail((2000, 2000))
        processed_images.append(img)

    # Build multi-image chat message
    content = []
    for img in processed_images:
        image_uri = pil_to_data_uri(img)
        content.append({"type": "image_url", "image_url": {"url": image_uri}})
    content.append({"type": "text", "text": SYSTEM_PROMPT})

    messages = [{"role": "user", "content": content}]

    try:
        outputs = llm.chat(messages, sampling_params=sampling_params)
        text = outputs[0].outputs[0].text
        return [text]  # Single combined output for all pages

    except Exception as e:
        log(f"Inference error: {e}")
        return [json.dumps({"error": f"Batch failed: {str(e)}"})]


def parse_raw_output(raw_output, batch_idx):
    """Parse raw model output into transaction list."""
    try:
        cleaned = raw_output

        # Strip markdown code fences
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # Try direct JSON parse first
        batch_data = None
        try:
            batch_data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback 1: extract JSON array using regex
            json_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if json_match:
                try:
                    batch_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Fallback 2: repair truncated JSON (model ran out of tokens)
            if batch_data is None:
                log("Direct parse failed. Attempting truncated JSON repair...")
                batch_data = repair_truncated_json(cleaned)

        if batch_data is not None and isinstance(batch_data, list):
            log(f"Batch {batch_idx} parsed successfully: {len(batch_data)} transactions.")
            return batch_data
        elif batch_data is not None:
            log(f"Warning: Batch {batch_idx} returned non-list JSON: {batch_data}")
        else:
            log(f"Failed to parse JSON for batch {batch_idx}. Skipping.")
            log(f"Raw output (first 500 chars): {raw_output[:500]}")
            log(f"Cleaned text (first 300 chars): {cleaned[:300]}")
    except Exception as e:
        log(f"Failed to parse JSON for batch {batch_idx}: {e}. Skipping.")
        log(f"Raw output (first 500 chars): {raw_output[:500]}")
    
    return []


def process_pdf(pdf_bytes):
    # 1. Convert PDF to Images
    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        log(f"Converted PDF to {len(images)} images.")
    except Exception as e:
        log(f"Error converting PDF: {e}")
        return json.dumps({"error": f"Failed to convert PDF: {str(e)}"})

    if not images:
        return json.dumps({"error": "No images extracted from PDF"})

    all_transactions = []
    
    # Process in batches — Qwen3-VL handles multi-image natively
    for i in range(0, len(images), MAX_PAGES_PER_BATCH):
        batch = images[i:i + MAX_PAGES_PER_BATCH]
        batch_num = i // MAX_PAGES_PER_BATCH + 1
        total_batches = (len(images) + MAX_PAGES_PER_BATCH - 1) // MAX_PAGES_PER_BATCH
        log(f"Processing batch {batch_num}/{total_batches} ({len(batch)} pages as multi-image prompt)...")
        
        # Process all pages in the batch as a single multi-image prompt
        raw_outputs = process_pages(batch)
        
        # Parse the combined output
        for j, raw_output in enumerate(raw_outputs):
            batch_transactions = parse_raw_output(raw_output, batch_num)
            all_transactions.extend(batch_transactions)
            
    # Filter out ghost transactions (balance=0, credit=null, debit=null)
    final_transactions = []
    for t in all_transactions:
        balance = t.get("balance")
        credit = t.get("credit")
        debit = t.get("debit")
        
        # Check if it's a ghost record
        if ((balance == 0 or balance == 0.0) and credit is None and debit is None) or (credit is None and debit is None) or (credit == 0 and debit == 0):
            continue
        
        # Keep only the 6 required fields
        cleaned_t = {
            "date": t.get("date", ""),
            "description": t.get("description", ""),
            "debit": t.get("debit"),
            "credit": t.get("credit"),
            "balance": t.get("balance"),
            "currency": t.get("currency", "")
        }
        final_transactions.append(cleaned_t)
    
    # ---- Post-processing: normalize dates (fix DD/MM vs MM/DD ambiguity) ----
    for t in final_transactions:
        date_str = t.get("date", "")
        if date_str:
            try:
                parts = date_str.split("-")
                if len(parts) == 3:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    if month > 12 and day <= 12:
                        log(f"Fixing swapped date: {date_str} -> {year}-{day:02d}-{month:02d}")
                        t["date"] = f"{year}-{day:02d}-{month:02d}"
            except (ValueError, IndexError):
                pass
    
    # ---- Post-processing: sort by date (chronological) for balance validation ----
    try:
        final_transactions.sort(key=lambda t: t.get("date", "0000-00-00"))
    except Exception:
        pass
    
    # ---- Post-processing: validate debit/credit using balance changes ----
    for i in range(1, len(final_transactions)):
        prev_balance = final_transactions[i - 1].get("balance")
        curr_balance = final_transactions[i].get("balance")
        credit = final_transactions[i].get("credit")
        debit = final_transactions[i].get("debit")
        
        if prev_balance is None or curr_balance is None:
            continue
        
        balance_diff = curr_balance - prev_balance
        
        if balance_diff < 0:
            if credit is not None and debit is None:
                log(f"Correcting transaction {i}: credit -> debit (balance decreased by {abs(balance_diff):.2f})")
                final_transactions[i]["debit"] = credit
                final_transactions[i]["credit"] = None
        
        elif balance_diff > 0:
            if debit is not None and credit is None:
                log(f"Correcting transaction {i}: debit -> credit (balance increased by {balance_diff:.2f})")
                final_transactions[i]["credit"] = debit
                final_transactions[i]["debit"] = None
            
    return final_transactions

# ===============================
# RUNPOD HANDLER
# ===============================
def handler(event):
    log(f"Received event: {event.keys()}")
    if "input" not in event:
        log("ERROR: No 'input' key in event")
        return {"error": "No input provided"}
        
    job_input = event["input"]
    log(f"Input keys: {job_input.keys() if isinstance(job_input, dict) else type(job_input)}")
    
    # Accept either 'pdf_base64' or 'file' as the input key
    pdf_b64 = job_input.get("pdf_base64") or job_input.get("file")
    
    if not pdf_b64:
        log("ERROR: Missing 'pdf_base64' or 'file' in input")
        return {"error": "Missing pdf_base64 or file field"}

    log(f"Received PDF data of length: {len(pdf_b64)}")
    
    try:
        pdf_bytes = base64.b64decode(pdf_b64)
        log(f"Decoded PDF: {len(pdf_bytes)} bytes")
    except Exception as e:
        log(f"ERROR: Invalid base64: {str(e)}")
        return {"error": f"Invalid base64: {str(e)}"}

    # Run Inference
    try:
        final_data = process_pdf(pdf_bytes)
        log(f"Processing complete. Transactions found: {len(final_data) if isinstance(final_data, list) else 'N/A'}")
        return final_data
    except Exception as e:
        log(f"ERROR during process_pdf: {str(e)}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Processing failed: {str(e)}"}

if __name__ == "__main__":
    # Log GPU status at startup
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        log("WARNING: CUDA is NOT available! Model will run on CPU (very slow).")
    runpod.serverless.start({"handler": handler})
