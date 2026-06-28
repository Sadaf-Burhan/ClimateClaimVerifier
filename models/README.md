# Models — Week 5 LoRA adapter (demo)

The production classifier is the **base `qwen2.5:3b`** (recall-first, single-post). The
LoRA adapter here is **demo-only** — it powers the "Base vs Adapter" tab so the
precision/recall tradeoff is visible, but it is not the deployed model.

## Register the adapter in Ollama (one-time)

1. **On Colab**, after training, export the merged adapter to GGUF (see the Week-5
   notebook / project README) and download it.
2. Put the file here as `models/climate_claim.gguf` (gitignored — ~2 GB).
3. From the repo root:
   ```bash
   ollama create qwen2.5-3b-claim-lora -f models/Modelfile
   ```
4. Confirm: `ollama list` shows `qwen2.5-3b-claim-lora`. The app's adapter column now runs.

The model name `qwen2.5-3b-claim-lora` must match `model.adapter_name` in
`src/climate_verifier/config.yaml`.

Note on quantization: q4_k_m (~1.9 GB) drifts marginally from the fp16 adapter on
borderline cases — it flipped the denial-with-statistic acceptance test ("50,000 acres…")
to opinion. **q8_0 (~3.3 GB) is near-lossless and matches the fp16 adapter on all
acceptance cases** (the 50k post stays `claim`), so it's the recommended export for the
demo. The authoritative metrics are always the fp16 eval (recall 0.854 / precision 0.788).
