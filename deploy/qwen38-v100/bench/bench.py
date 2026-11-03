import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LLAMA_REPO", os.path.abspath(os.path.join(HERE, "../../..")))
URL = os.environ.get("QWEN38_URL", "http://127.0.0.1:18080")
TESTIMG = os.environ.get("QWEN38_TESTIMG", os.path.join(HERE, "testimg"))
OUT = os.environ.get("QWEN38_OUT", os.getcwd())
import json, sys, urllib.request, statistics
P = ["Write a 600-word short story about a lighthouse keeper who finds a strange radio signal.",
     "Explain in detail how TCP congestion control works, including slow start, AIMD, and fast recovery.",
     "Compare the economic policies of three different historical periods of your choice, in about 600 words.",
     "Write a Rust implementation of a thread-safe LRU cache with tests.",
     "Describe the history of the printing press and its effects on European society."]
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1
tot_n = tot_ms = tot_d = tot_a = 0
for r in range(reps):
  for i, p in enumerate(P):
    body = json.dumps({"messages":[{"role":"user","content":p}],"max_tokens":800,"seed":1000+r*10+i,
                       "chat_template_kwargs":{"enable_thinking":False}}).encode()
    t = json.load(urllib.request.urlopen(urllib.request.Request(URL+"/v1/chat/completions", body, {"Content-Type":"application/json"}), timeout=600))["timings"]
    tot_n += t["predicted_n"]; tot_ms += t["predicted_ms"]; tot_d += t.get("draft_n",0); tot_a += t.get("draft_n_accepted",0)
    print(f"  {t['predicted_per_second']:6.1f} t/s  acc {100*t.get('draft_n_accepted',0)/max(1,t.get('draft_n',1)):4.0f}%  n={t['predicted_n']}", flush=True)
acc = tot_a/max(1,tot_d); steps = tot_n - tot_a; ms = tot_ms/steps
print(f"OVERALL {1000*tot_n/tot_ms:.1f} t/s  acceptance {100*acc:.1f}%  ({tot_n} tok)  {ms:.2f} ms/step  tok/step {tot_n/steps:.2f}  -> at 50%: {2500/ms:.1f} t/s")
