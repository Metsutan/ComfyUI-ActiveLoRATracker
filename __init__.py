import asyncio
import hashlib
import json
import os
import re
import shutil
import struct
import urllib.request
from html.parser import HTMLParser
from aiohttp import web
import folder_paths
from server import PromptServer

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

OLLAMA_BASE = os.environ.get("OLLAMA_INTERNAL_URL", "http://127.0.0.1:11434")
CIVITAI_TOKEN = os.environ.get("CIVITAI_API_TOKEN", "")

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_GUIDE_PATH = os.path.join(NODE_DIR, "H3_ref_en.md")
TMP_GUIDE_PATH = "/tmp/h3_setup/H3_ref_en.md"
OFFICIAL_GUIDE_URL = (
    "https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md"
)

CACHE_FILE = os.path.join(folder_paths.models_dir, "loras", ".civitai_cache.json")
SKIP_KEYWORDS = ("turbo", "silu_temb", "pdd_8step", "lightx2v", "rank8")

# Default prompt mode controlled by env var ("1", "true", "official" -> True)
_USE_OFFICIAL_GUIDE = os.environ.get("USE_OFFICIAL_H3_GUIDE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "official",
)


class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return re.sub(r"\s+", " ", "".join(self.fed)).strip()


def strip_html(text: str) -> str:
    if not text:
        return ""
    s = HTMLStripper()
    s.feed(text)
    return s.get_data()


def ensure_official_guide() -> str:
    """Ensures H3_ref_en.md is saved locally on disk and returns its contents."""
    if not os.path.exists(LOCAL_GUIDE_PATH) or os.path.getsize(LOCAL_GUIDE_PATH) < 1000:
        if os.path.exists(TMP_GUIDE_PATH) and os.path.getsize(TMP_GUIDE_PATH) > 1000:
            shutil.copyfile(TMP_GUIDE_PATH, LOCAL_GUIDE_PATH)
        else:
            print(f"[ActiveLoRATracker] Downloading official H3_ref_en.md to {LOCAL_GUIDE_PATH}...")
            req = urllib.request.Request(
                OFFICIAL_GUIDE_URL, headers={"User-Agent": "ComfyUI-ActiveLoRATracker/1.0"}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read().decode("utf-8")
            with open(LOCAL_GUIDE_PATH, "w", encoding="utf-8") as f:
                f.write(content)
            return content

    with open(LOCAL_GUIDE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def fetch_json(url: str) -> dict:
    headers = {"User-Agent": "ComfyUI-ActiveLoRATracker/1.0", "Content-Type": "application/json"}
    if CIVITAI_TOKEN and CIVITAI_TOKEN != "token":
        headers["Authorization"] = f"Bearer {CIVITAI_TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=12) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_safetensors_tags(filepath: str, top_n: int = 12) -> list:
    """Fallback: read top training tags from .safetensors header if Civitai has none."""
    try:
        with open(filepath, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            if header_size > 100_000_000:
                return []
            header = json.loads(f.read(header_size).decode("utf-8", errors="ignore"))
            tag_freq = header.get("__metadata__", {}).get("ss_tag_frequency")
            if tag_freq:
                counts = {}
                for ds in json.loads(tag_freq).values():
                    if isinstance(ds, dict):
                        for tag, cnt in ds.items():
                            counts[tag.strip()] = counts.get(tag.strip(), 0) + int(cnt)
                return [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    except Exception:
        pass
    return []


def resolve_lora_metadata(lora_filename: str) -> dict | None:
    if any(k in lora_filename.lower() for k in SKIP_KEYWORDS):
        return None

    full_path = folder_paths.get_full_path("loras", lora_filename)
    if not full_path or not os.path.exists(full_path):
        return None

    cache = load_cache()
    mtime = os.path.getmtime(full_path)
    if lora_filename in cache and cache[lora_filename].get("mtime") == mtime:
        return cache[lora_filename]

    print(f"[ActiveLoRATracker] Hashing & querying Civitai for active LoRA: {lora_filename}")
    h = hashlib.sha256()
    with open(full_path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    sha256 = h.hexdigest().upper()

    info = {
        "file": lora_filename,
        "mtime": mtime,
        "sha256": sha256,
        "title": lora_filename,
        "triggers": [],
        "description": "",
        "sample_prompts": [],
    }

    try:
        ver = fetch_json(f"https://civitai.com/api/v1/model-versions/by-hash/{sha256}")
        model_id = ver.get("modelId")
        ver_id = ver.get("id")
        info["title"] = f"{ver.get('model', {}).get('name', lora_filename)} ({ver.get('name', '')})"
        info["triggers"] = ver.get("trainedWords") or []
        ver_desc = strip_html(ver.get("description") or "")

        parent_desc = ""
        if model_id:
            try:
                mod = fetch_json(f"https://civitai.com/api/v1/models/{model_id}")
                parent_desc = strip_html(mod.get("description") or "")
            except Exception:
                pass
        info["description"] = " | ".join(filter(None, [parent_desc[:600], ver_desc[:400]]))

        prompts = []
        for img in ver.get("images") or []:
            meta = img.get("meta") or {}
            p = meta.get("prompt") or meta.get("Prompt") or ""
            if p and len(p) > 15 and p not in prompts:
                prompts.append(re.sub(r"\s+", " ", p).strip()[:450])
            if len(prompts) >= 2:
                break

        if not prompts and ver_id:
            try:
                feed = fetch_json(
                    f"https://civitai.com/api/v1/images?modelVersionId={ver_id}&withMeta=true&limit=10"
                )
                for item in feed.get("items") or []:
                    meta = item.get("meta") or {}
                    p = meta.get("prompt") or ""
                    if p and len(p) > 15 and p not in prompts:
                        prompts.append(re.sub(r"\s+", " ", p).strip()[:450])
                    if len(prompts) >= 2:
                        break
            except Exception:
                pass
        info["sample_prompts"] = prompts
    except Exception as e:
        print(f"[ActiveLoRATracker] Civitai lookup skipped for {lora_filename}: {e}")

    if not info["triggers"]:
        info["triggers"] = extract_safetensors_tags(full_path)

    cache[lora_filename] = info
    save_cache(cache)
    return info


def get_running_workflow_loras() -> list:
    """Reads the currently executing ComfyUI graph (bypassed/muted nodes are excluded)."""
    q = PromptServer.instance.prompt_queue
    running, pending = q.get_current_queue()
    graph = None
    if running:
        graph = running[0][2]
    elif pending:
        graph = pending[0][2]
    else:
        hist = q.get_history(max_items=1)
        if hist:
            last_item = next(iter(hist.values()))
            if "prompt" in last_item and len(last_item["prompt"]) > 2:
                graph = last_item["prompt"][2]

    if not isinstance(graph, dict):
        return []

    found = []
    for node in graph.values():
        inputs = node.get("inputs", {})
        for k, v in inputs.items():
            # Standard LoraLoader, LoraLoaderModelOnly, KJNodes, etc.
            if isinstance(v, str) and v.endswith(".safetensors") and "lora" in k.lower():
                suffix = k.replace("lora_name", "").replace("lora", "")
                strength = inputs.get(
                    f"strength_model{suffix}",
                    inputs.get(f"strength{suffix}", inputs.get("strength_model", 1.0)),
                )
                enabled = inputs.get(f"switch{suffix}", inputs.get("enabled", True))
                if enabled not in (False, "Off") and float(strength) != 0.0:
                    found.append({"name": v, "strength": float(strength)})
            # rgthree Power Lora Loader dictionary inputs
            elif isinstance(v, dict) and v.get("lora") and v.get("lora") != "None":
                strength = float(v.get("strength", v.get("strengthTwo", 1.0)))
                if v.get("on", True) and strength != 0.0:
                    found.append({"name": v["lora"], "strength": strength})
    return found


def build_enrichment_block() -> tuple[list, str]:
    active = get_running_workflow_loras()
    seen = set()
    blocks = []

    for item in active:
        name = item["name"]
        if name in seen:
            continue
        seen.add(name)
        meta = resolve_lora_metadata(name)
        if not meta:
            continue

        lines = [f"### Active LoRA: `{meta['title']}` (Strength: {item['strength']})"]
        if meta["triggers"]:
            lines.append(f"- **Required Trigger Words / Tags:** {', '.join(meta['triggers'])}")
        if meta["description"]:
            lines.append(f"- **Creator Instructions:** {meta['description']}")
        for idx, sp in enumerate(meta["sample_prompts"], 1):
            lines.append(f"- **Creator Showcase Prompt {idx}:** \"{sp}\"")
        blocks.append("\n".join(lines))

    base_rules = (
        "\n\n[CRITICAL H3 RULES]\n"
        "- Never emit <tool_call> or </tool_call> tags or markdown code fences. Output plain text only.\n"
        "- ISOLATED REFERENCE CROPS: If any input reference image shows a subject on a solid neutral grey (#B4B4B4), "
        "white, or blank background, extract ONLY the character/attire as <Subject N>. NEVER mention or describe "
        "the grey/blank studio backdrop in `subject_definitions` or `detailed_description`."
    )

    if not blocks:
        return active, base_rules

    lora_section = (
        "\n\n[ACTIVE WORKFLOW LORAS]\n"
        "The following LoRAs are active in this generation. Weave their Required Trigger Words, Creator Instructions, "
        "and Showcase Prompt phrasing directly into `subject_definitions` and `detailed_description`:\n\n"
        + "\n\n".join(blocks)
    )
    return active, base_rules + lora_section


def build_official_system_prompt(enrichment: str) -> str:
    guide_text = ensure_official_guide()
    header = (
        "You are MiniMax H3-Context-IR. Output ONLY the final structured 6-section MiniMax H3 production prompt—"
        "no conversational preamble and no markdown code fences.\n"
        "NOTE ON LOCAL VIDEO REFERENCES: When <Video N> inputs are provided as sampled frame sequences without "
        "attached audio tracks, infer motion and camera rhythm from the frame progression.\n\n"
    )
    return header + guide_text + enrichment


def _forward_to_ollama(endpoint: str, method: str = "GET", payload: bytes | None = None) -> bytes:
    url = f"{OLLAMA_BASE.rstrip('/')}{endpoint}"
    headers = {"Content-Type": "application/json"} if payload else {}
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


async def _handle_chat_completions(request: web.Request, force_official: bool | None = None) -> web.Response:
    body = await request.json()
    use_official = _USE_OFFICIAL_GUIDE if force_official is None else force_official

    active, enrichment = await asyncio.to_thread(build_enrichment_block)
    messages = body.get("messages", [])

    if messages:
        sys_msg = next((m for m in messages if m.get("role") == "system"), None)
        if use_official:
            official_prompt = await asyncio.to_thread(build_official_system_prompt, enrichment)
            if sys_msg:
                sys_msg["content"] = official_prompt
            else:
                messages.insert(0, {"role": "system", "content": official_prompt})
            print(f"[ActiveLoRATracker] Mode: OFFICIAL H3_ref_en.md | Active LoRAs: {len(active)}")
        else:
            if sys_msg and isinstance(sys_msg.get("content"), str):
                sys_msg["content"] += enrichment
            else:
                messages.insert(0, {"role": "system", "content": enrichment.strip()})
            print(f"[ActiveLoRATracker] Mode: MINIMAX_REFPACK BUILT-IN | Active LoRAs: {len(active)}")

    raw_resp = await asyncio.to_thread(
        _forward_to_ollama,
        "/v1/chat/completions",
        "POST",
        json.dumps(body).encode("utf-8"),
    )
    return web.Response(body=raw_resp, content_type="application/json")


# Pre-cache H3_ref_en.md on startup in a background thread so first run never lags
asyncio.get_event_loop().run_in_executor(None, ensure_official_guide)


@PromptServer.instance.routes.get("/active_loras/context")
async def get_active_loras_debug(request):
    active, enrichment = await asyncio.to_thread(build_enrichment_block)
    return web.json_response(
        {
            "default_mode": "official" if _USE_OFFICIAL_GUIDE else "refpack",
            "local_guide_cached": os.path.exists(LOCAL_GUIDE_PATH),
            "active_loras": active,
            "enrichment": enrichment,
        }
    )


@PromptServer.instance.routes.get("/active_loras/mode/{mode}")
async def set_default_prompt_mode(request):
    global _USE_OFFICIAL_GUIDE
    mode = request.match_info.get("mode", "").lower()
    if mode in ("official", "1", "true"):
        _USE_OFFICIAL_GUIDE = True
    elif mode in ("refpack", "builtin", "0", "false"):
        _USE_OFFICIAL_GUIDE = False
    else:
        return web.json_response({"error": "Use /active_loras/mode/official or /active_loras/mode/refpack"}, status=400)
    return web.json_response({"default_mode": "official" if _USE_OFFICIAL_GUIDE else "refpack"})


# Standard routes (follow _USE_OFFICIAL_GUIDE default)
@PromptServer.instance.routes.get("/active_loras/v1/models")
@PromptServer.instance.routes.get("/active_loras/official/v1/models")
@PromptServer.instance.routes.get("/active_loras/refpack/v1/models")
async def proxy_models(request):
    raw = await asyncio.to_thread(_forward_to_ollama, "/v1/models", "GET", None)
    return web.Response(body=raw, content_type="application/json")


@PromptServer.instance.routes.post("/active_loras/v1/chat/completions")
async def proxy_chat_default(request):
    return await _handle_chat_completions(request, force_official=None)


# Explicit URL-based mode overrides (switch directly inside the MiniMaxRefPack node's api_base widget)
@PromptServer.instance.routes.post("/active_loras/official/v1/chat/completions")
async def proxy_chat_official(request):
    return await _handle_chat_completions(request, force_official=True)


@PromptServer.instance.routes.post("/active_loras/refpack/v1/chat/completions")
async def proxy_chat_refpack(request):
    return await _handle_chat_completions(request, force_official=False)
