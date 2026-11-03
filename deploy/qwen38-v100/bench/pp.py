import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LLAMA_REPO", os.path.abspath(os.path.join(HERE, "../../..")))
URL = os.environ.get("QWEN38_URL", "http://127.0.0.1:18080")
TESTIMG = os.environ.get("QWEN38_TESTIMG", os.path.join(HERE, "testimg"))
OUT = os.environ.get("QWEN38_OUT", os.getcwd())
import json, sys, glob, urllib.request
src = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob(sys.argv[1] + "/**/*.cpp", recursive=True)))
off = 0
for target in [2000, 8000, 32000, 64000, 128000]:
    chars = int(target * 3.3)
    text = src[off:off + chars]; off += chars
    body = json.dumps({"prompt": text, "n_predict": 1, "cache_prompt": True}).encode()
    req = urllib.request.Request(URL+"/completion", body, {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=1800))["timings"]
    print(f"{t['prompt_n']:>7} tok  {t['prompt_per_second']:8.1f} t/s  {t['prompt_ms']/1000:6.1f} s", flush=True)
print("source chars used", off, "of", len(src))
