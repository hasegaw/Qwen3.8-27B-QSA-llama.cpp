import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LLAMA_REPO", os.path.abspath(os.path.join(HERE, "../../..")))
URL = os.environ.get("QWEN38_URL", "http://127.0.0.1:18080")
TESTIMG = os.environ.get("QWEN38_TESTIMG", os.path.join(HERE, "testimg"))
OUT = os.environ.get("QWEN38_OUT", os.getcwd())
import json, glob, time, threading, base64, urllib.request, subprocess, sys
L = REPO
docs = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob(L+"/docs/**/*.md", recursive=True)))
docs += "".join(open(f, errors="ignore").read() for f in sorted(glob.glob(L+"/tools/**/*.md", recursive=True)))
pid = subprocess.run(["pgrep","-x","llama-server"], capture_output=True, text=True).stdout.split()[0]
peak = {"anon": 0, "file": 0}; stop = False
def rd():
    d = {}
    for l in open(f"/proc/{pid}/status"):
        if l.startswith(("RssAnon", "RssFile")): d[l.split(":")[0]] = int(l.split()[1]) / 1048576
    return d
def sampler():
    while not stop:
        d = rd(); peak["anon"] = max(peak["anon"], d["RssAnon"]); peak["file"] = max(peak["file"], d["RssFile"]); time.sleep(0.2)
threading.Thread(target=sampler, daemon=True).start()
def chat(msgs, mt=60):
    body = {"messages": msgs, "max_tokens": mt, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); j = json.load(urllib.request.urlopen(urllib.request.Request(URL+"/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"}), timeout=3600))
    return j["timings"], time.time() - t0
def step(name, msgs):
    peak["anon"] = 0; t, dt = chat(msgs); d = rd()
    print(f"{name:34s} prefilled {t['prompt_n']:6d} reused {t.get('cache_n',0):6d} {dt:6.1f}s | decode {t['predicted_per_second']:5.1f} t/s | anon now {d['RssAnon']:5.1f} GB peak {peak['anon']:5.1f} GB | file {d['RssFile']:5.1f} GB", flush=True)
U = lambda c: {"role": "user", "content": c}
A = [U("Document:\n\n" + docs[:480000] + "\n\nWhat is this about? One sentence.")]
B = [U("Document:\n\n" + docs[500000:620000] + "\n\nSummarize in one sentence.")]
C = [U("Document:\n\n" + docs[640000:760000] + "\n\nSummarize in one sentence.")]
img = base64.b64encode(open(os.path.join(TESTIMG, "large.png"),"rb").read()).decode()
step("A: 142K-token doc (cold)", A)
step("B: 34K doc (A saved to cache)", B)
step("C: 34K doc (B saved to cache)", C)
step("A again (restore from cache)", A)
step("B again (restore from cache)", B)
step("image request", [{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"+img}},{"type":"text","text":"What number is shown?"}]}])
stop = True
print(f"overall peak anon {max(peak['anon'], rd()['RssAnon']):.1f} GB")
