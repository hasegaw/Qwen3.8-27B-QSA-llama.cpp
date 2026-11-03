import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LLAMA_REPO", os.path.abspath(os.path.join(HERE, "../../..")))
URL = os.environ.get("QWEN38_URL", "http://127.0.0.1:18080")
TESTIMG = os.environ.get("QWEN38_TESTIMG", os.path.join(HERE, "testimg"))
OUT = os.environ.get("QWEN38_OUT", os.getcwd())
import json, glob, time, base64, urllib.request
L = REPO
docs = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob(L+"/docs/**/*.md", recursive=True)))
docA, docB = docs[:90000], docs[200000:290000]   # ~25K tokens each, different content
img = base64.b64encode(open(os.path.join(TESTIMG, "large.png"),"rb").read()).decode()
def chat(msgs, max_tokens=60):
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    j = json.load(urllib.request.urlopen(urllib.request.Request(URL+"/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"}), timeout=1800))
    t = j["timings"]; return j["choices"][0]["message"]["content"], t, time.time() - t0
def row(name, t, dt):
    tot = t.get("cache_n", 0) + t["prompt_n"]
    print(f"{name:44s} prompt {tot:6d} tok | reused {t.get('cache_n',0):6d} | prefilled {t['prompt_n']:6d} in {t['prompt_ms']/1000:5.2f}s | total {dt:5.2f}s", flush=True)
U = lambda c: {"role": "user", "content": c}; A = lambda c: {"role": "assistant", "content": c}
# 1. multi-turn conversation over a long document (document in the first user message)
conv = [U("Here is a document:\n\n" + docA + "\n\nWhat is it about? One sentence.")]
a, t, dt = chat(conv); row("1a first turn (doc + question)", t, dt)
conv += [A(a), U("List three section titles from it.")]
a, t, dt = chat(conv); row("1b next turn (append)", t, dt)
conv += [A(a), U("Which build options are mentioned?")]
a, t, dt = chat(conv); row("1c next turn (append)", t, dt)
# 2. regenerate the same last turn
a2, t, dt = chat(conv); row("2  regenerate last answer", t, dt)
# 3. edit the last user message
conv[-1] = U("Which GPU backends are mentioned?")
a, t, dt = chat(conv); row("3  edit last question", t, dt)
# 4. switch to another conversation and back
other = [U("Here is another document:\n\n" + docB + "\n\nSummarize it in one sentence.")]
a, t, dt = chat(other); row("4a switch to conversation B (new doc)", t, dt)
a, t, dt = chat(conv); row("4b back to conversation A", t, dt)
# 5. same doc, different question in the SAME message
s1 = [U("Here is a document:\n\n" + docB + "\n\nQuestion: what tools does it describe?")]
a, t, dt = chat(s1); row("5a doc+question in one message", t, dt)
s2 = [U("Here is a document:\n\n" + docB + "\n\nQuestion: what file formats does it mention?")]
a, t, dt = chat(s2); row("5b same doc, different question (same msg)", t, dt)
# 6. doc as an earlier message, question as a new message
d = [U("Here is a document:\n\n" + docB), A("Got it."), U("What tools does it describe?")]
a, t, dt = chat(d); row("6a doc in its own message + question", t, dt)
d[-1] = U("What file formats does it mention?")
a, t, dt = chat(d); row("6b different question (separate msg)", t, dt)
# 7. image conversation: follow-up turn re-sends the same image
im = [{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"+img}},{"type":"text","text":"What number is in the image?"}]}]
a, t, dt = chat(im); row("7a image + question", t, dt)
im += [A(a), U("And what shapes?")]
a, t, dt = chat(im); row("7b follow-up turn (image in history)", t, dt)
