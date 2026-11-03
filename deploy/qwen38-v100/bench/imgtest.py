import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LLAMA_REPO", os.path.abspath(os.path.join(HERE, "../../..")))
URL = os.environ.get("QWEN38_URL", "http://127.0.0.1:18080")
TESTIMG = os.environ.get("QWEN38_TESTIMG", os.path.join(HERE, "testimg"))
OUT = os.environ.get("QWEN38_OUT", os.getcwd())
import json, base64, time, urllib.request, sys, subprocess
def ask(path, q="What 4-digit number is written in this image, and what colored shapes are there? Answer in one short sentence."):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    body = {"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}},{"type":"text","text":q}]}],
            "max_tokens":80,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"cache_prompt":False}
    t0 = time.time()
    j = json.load(urllib.request.urlopen(urllib.request.Request(URL+"/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"}), timeout=600))
    dt = time.time() - t0; t = j["timings"]; a = j["choices"][0]["message"]["content"].replace("\n", " ")
    ok = all(k in a.lower() for k in ["7391", "red", "blue", "green"])
    vram = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()
    print(f"{path.split('/')[-1]:10s} total {dt:5.2f}s  prompt {t['prompt_n']:5d} tok in {t['prompt_ms']/1000:5.2f}s  decode {t['predicted_per_second']:5.1f} t/s  {'OK ' if ok else 'BAD'}  vram {vram}  | {a[:90]}", flush=True)
for p in sys.argv[1:]:
    ask(p)
