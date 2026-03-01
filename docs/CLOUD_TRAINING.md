# Cloud Training — Best Results

After fine-tuning a model in the cloud with your warehouse/inventory data, you can use it in this app for more accurate, domain-specific answers. Two options are supported; once trained, you can select the model in the app.

---

## Do we put the project on the cloud to train?

**No.** You do **not** deploy or run this whole app on the cloud for training. Only the **training job** runs on the cloud (Vertex AI or Google AI Studio). Your code and repo stay local. You:

1. **Generate training data locally** (see below).
2. **Upload one file** — the JSONL dataset — to Vertex or AI Studio.
3. **Start a training job** in the cloud (pick base model, point to your JSONL).
4. When done, **download/export** the model (for offline) or use the tuned model ID (for Gemini API).

So: **train on cloud, data prepared locally.**

---

## Where is the training data generated?

**In this repo.** Use the script that generates prompt/response pairs from warehouse data. Training data is **not hardcoded** if you pass current data in:

- **Script:** `scripts/generate_training_data.py`
- **Input (no hardcoding):** Pass an inventory JSON exported from the app or your DB:
  ```bash
  python scripts/generate_training_data.py -o training_data.jsonl --inventory exported_inventory.json
  ```
  The script then runs `calculate_reorder` / `forecast_demand` on that data to build Q&A pairs. So the content comes from your real/current state.
- **Input (fallback):** If you omit `--inventory`, the script uses the default seed from `backend/data.py` (for quick local tests only).
- **Output:** A JSONL file, e.g. `training_data.jsonl`.

Run it locally:

```bash
# From project root — use real data (export inventory from app or DB to a JSON file first)
python scripts/generate_training_data.py -o training_data.jsonl --inventory exported_inventory.json

# Or with default seed only (hardcoded fallback)
python scripts/generate_training_data.py -o training_data.jsonl
```

Then upload `training_data.jsonl` to Vertex AI or AI Studio when you create the training job. To get `exported_inventory.json`, export current inventory from the app (e.g. Manager → Import/Export or a dedicated “Export for training” action that downloads `live_inventory` as JSON) or from your database.

**What the training data teaches:** The script generates data so the model learns **which action corresponds to which question**, not just numbers. Each response includes **Reasoning** (推理过程), **Action**, and **Result**, with six action types (including **recommend next purchase** — 推荐接下来购入什么产品, considering 销量、时间、urgency): **stock lookup**, **reorder check**, **location lookup**, **supplier lookup**, **demand forecast**. Multiple question phrasings are used per action (e.g. “Should we reorder X?”, “Do we need to order more X?”) so the model generalizes: same intent → same action and response structure.

**Runtime vs training data:** In the app, answers are driven by **live state**, not hardcoded data. When a user asks a question, the backend uses `st.session_state.live_inventory` and calls `calculate_reorder()`, `forecast_demand()`, etc. with that state. The JSONL trains the model to map question types to the right “action”; at inference time the app injects current data into the prompt and the model answers using that action pattern.

---

## Which option to use?

| Goal | Use this |
|------|----------|
| **Final model must run offline (on-device / no cloud at inference)** | **Option 2: Vertex AI + Gemma** — Train on Google Cloud (Vertex), then **export** the trained model (e.g. GGUF) and run it **locally with Ollama**. Inference is 100% offline. |
| Final model can stay in the cloud (call via API) | Option 1: AI Studio + Gemini — Easiest; tuned model is used via Gemini API. |

**If you have Google Cloud and need the model to be offline:** use **Option 2** below. Train on Vertex AI, export the checkpoint, convert to a format Ollama accepts, then run Ollama on your own machine — no internet required at inference time.

---

## Option 1: Google AI Studio — Fine-tune Gemini (simplest, model stays in cloud)

Best for: Keeping **Gemini API** and wiring the tuned model into the app quickly.

1. **Open AI Studio**  
   https://aistudio.google.com/ → **Fine-tuning** (or **Create tuned model**).

2. **Prepare training data**  
   JSONL format, one object per line, e.g.:
   ```json
   {"prompt": "What is current stock for CB-1201-A? Should we reorder?", "response": "Current stock 45, below safety. Recommend reordering 50 units."}
   ```
   Export warehouse FAQs, reorder advice, or cycle-count notes into 50–500+ examples.

3. **Pick base model**  
   Choose **Gemini 2.0 Flash** or **Gemini 1.5 Flash**, upload the JSONL, and start training.

4. **Get the Tuned Model name**  
   When training finishes you get a model ID like `tunedModels/xxx` or `models/xxx`.

5. **Use it in this app**  
   - In the sidebar select **☁️ Gemini (Cloud)** and enter your API Key.  
   - In **Cloud-trained model ID** paste the Tuned Model name (if empty, the app uses default `gemini-2.5-flash`).  
   - All Gemini-backed chat will then use your trained model for best results.

---

## Option 2: Vertex AI — Fine-tune Gemma → export for **offline** use (recommended for on-device)

**Use this when the model must run offline.** Train on Google Cloud (Vertex AI), then export and run with Ollama on your own hardware — no cloud at inference.

Best for: **Offline / on-device** deployment, Gemma, or full control over where the model runs.

1. **Setup**  
   - Enable **Vertex AI API** in your Google Cloud project.  
   - Install: `pip install google-cloud-aiplatform`

2. **Prepare data**  
   Same as above: JSONL with prompt/response pairs (or whatever format Vertex requires; see official docs).

3. **Run fine-tuning**  
   - Use the official notebook:  
     [Vertex AI TRL Fine-tuning Gemma](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/open-models/fine-tuning/vertex_ai_trl_fine_tuning_gemma.ipynb)  
   - Or docs:  
     [Gemma model fine-tuning](https://ai.google.dev/gemma/docs/tune)  
   - In Vertex console create a **Training job**, select a Gemma base model, upload data, and start.

4. **Export and run offline**  
   - **For offline:** Export the trained model (e.g. to GGUF / Safetensors), then convert and load in **Ollama** on your machine. In this app, choose **🔒 Local model** and select your custom model name — inference stays fully offline.  
   - (Optional) **A**: Alternatively deploy as an online Endpoint on Vertex and call it from the app via Vertex API if you ever want a cloud option.

5. **Wire Vertex endpoint into this app (optional)**  
   If deployed on Vertex, set in `.env` or config:
   - `VERTEX_PROJECT`
   - `VERTEX_LOCATION`
   - `VERTEX_ENDPOINT_ID`  
   and enable “cloud-trained model” mode in the backend so requests go to that endpoint instead of default Gemini.

---

## Cloud-trained model in this app

- **Gemini users**: In the sidebar/settings, set **Cloud-trained model ID** to the Tuned Model name from AI Studio to use your trained model for best results.  
- **Vertex users**: After configuring project and endpoint, enable “Use Vertex cloud-trained model” so requests are sent to your deployed model on Vertex.

See in-app **Settings** or the sidebar **Cloud-trained model ID** for details.

---

## Using a fine-tuned model from a compressed package (zip)

If you exported your fine-tuned model into a **zip** (or tar) and want to run it **locally with Ollama** in this app:

### 1. Extract and import into Ollama

- **Extract** the zip to a folder (e.g. `my-model/`).
- If the package contains a **Modelfile** (and optionally GGUF weights):
  ```bash
  cd my-model
  ollama create -f Modelfile
  ```
  This creates a model with the name defined in the Modelfile (e.g. `FROM ./model.gguf` and the name you give).
- If the package is an **Ollama model bundle** (e.g. a single file or folder Ollama expects), follow the export tool's instructions (e.g. `ollama create my-model -f Modelfile` or copying into Ollama's model directory, depending on the tool you used).
- Confirm the model is available:
  ```bash
  ollama list
  ```
  Note the **exact model name** (e.g. `warehouse-assistant` or `my-warehouse:latest`).

### 2. Use it in this app

1. In the sidebar, choose **Local (Ollama)**.
2. In **"Or use fine-tuned model (from zip)"**, enter the **exact model name** from `ollama list` (e.g. `warehouse-assistant`).
3. Leave the dropdown as-is; the app will use the name you typed for all Ollama calls.
4. Ask your questions; the app will use your fine-tuned model for answers.

If the name is wrong or the model is not loaded, Ollama may error and the app will fall back to the rule engine. Fix the name or run `ollama run <model-name>` once to load it, then try again.
