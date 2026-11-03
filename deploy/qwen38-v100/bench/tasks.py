import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LLAMA_REPO", os.path.abspath(os.path.join(HERE, "../../..")))
URL = os.environ.get("QWEN38_URL", "http://127.0.0.1:18080")
TESTIMG = os.environ.get("QWEN38_TESTIMG", os.path.join(HERE, "testimg"))
OUT = os.environ.get("QWEN38_OUT", os.getcwd())
import json, sys, glob, urllib.request, time
U = URL
def chat(prompt, max_tokens, seed, think=False):
    body = {"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"seed":seed,
            "chat_template_kwargs":{"enable_thinking":think}}
    req = urllib.request.Request(U+"/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"})
    j = json.load(urllib.request.urlopen(req, timeout=1800))
    m = j["choices"][0]["message"]
    return j["timings"], (m.get("content") or ""), (m.get("reasoning_content") or "")
L = REPO
readme = open(L+"/README.md").read()
sampling_h = open(L+"/common/sampling.h").read()
article = readme[:12000]
passage = readme[2000:5000]
docs = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob(L+"/docs/**/*.md", recursive=True)))
docs += "".join(open(f, errors="ignore").read() for f in sorted(glob.glob(L+"/tools/**/*.md", recursive=True)))
def needle_doc(chars):
    d = docs[:chars]; mid = len(d)//2
    return d[:mid] + "\n\nNote for reviewers: the internal codename of this release is BLUE-HERON-42.\n\n" + d[mid:]
T = [
 ("prose",       "Write a 500-word short story about a lighthouse keeper who finds a strange radio signal.", 900, False, None),
 ("explain",     "Explain in detail how TCP congestion control works, including slow start, AIMD, and fast recovery.", 900, False, None),
 ("codegen",     "Write a Python implementation of a thread-safe LRU cache class with type hints, docstrings and pytest tests.", 1200, False, None),
 ("code-edit",   "Rename `common_sampler_sample` to `common_sampler_draw` everywhere in this file. Output the complete updated file in one ```cpp block and nothing else.\n\n```cpp\n"+sampling_h+"```", 2500, False, None),
 ("summarize",   "Summarize the following document in about 200 words, as bullet points.\n\n"+article, 600, False, None),
 ("translate",   "Translate the following text into Japanese. Keep code and command names unchanged.\n\n"+passage, 1500, False, None),
 ("json",        "Extract every command-line tool, model format and hardware backend mentioned in the text below as JSON with keys tools, formats, backends (arrays of strings). Output only JSON.\n\n"+article, 700, False, None),
 ("japanese",    "日本の四季と、それぞれの季節の代表的な行事や食べ物について、詳しく説明してください。", 900, False, None),
 ("math-think",  "A tank is filled by pipe A in 6 hours and by pipe B in 9 hours, and drained by pipe C in 12 hours. If all three are opened at 8:00 with the tank empty, at what time is it full? Show your reasoning and give the exact time.", 3000, True, None),
 ("longctx-32k", "Read the document and answer: what is the internal codename of this release? Then list three topics the document covers.\n\n"+needle_doc(120000), 300, False, "BLUE-HERON-42"),
 ("longctx-128k","Read the document and answer: what is the internal codename of this release? Then list three topics the document covers.\n\n"+needle_doc(480000), 300, False, "BLUE-HERON-42"),
]
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
res = []
for name, p, mt, think, check in T:
    if only and name not in only: continue
    nrep = 1 if name.startswith("longctx") else reps
    for r in range(nrep):
        t, out, rc = chat(p, mt, 100 + r, think)
        d, a = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        ok = "" if not check else ("ok" if check in out else "MISS")
        row = dict(task=name, rep=r, prompt_n=t["prompt_n"], prefill_tps=t["prompt_per_second"], prefill_s=t["prompt_ms"]/1000,
                   gen_n=t["predicted_n"], decode_tps=t["predicted_per_second"], acc=(a/d if d else 0), check=ok, think_chars=len(rc))
        res.append(row)
        print(f"{name:13s} r{r}  prompt {t['prompt_n']:6d} tok @ {t['prompt_per_second']:7.1f} t/s ({t['prompt_ms']/1000:5.1f}s) | gen {t['predicted_n']:5d} tok @ {t['predicted_per_second']:6.1f} t/s | accept {100*(a/d if d else 0):4.0f}% {ok}", flush=True)
json.dump(res, open(sys.argv[3] if len(sys.argv) > 3 else os.path.join(OUT, "task-bench.json"), "w"), indent=1)
