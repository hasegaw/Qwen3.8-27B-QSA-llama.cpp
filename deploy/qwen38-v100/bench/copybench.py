import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LLAMA_REPO", os.path.abspath(os.path.join(HERE, "../../..")))
URL = os.environ.get("QWEN38_URL", "http://127.0.0.1:18080")
TESTIMG = os.environ.get("QWEN38_TESTIMG", os.path.join(HERE, "testimg"))
OUT = os.environ.get("QWEN38_OUT", os.getcwd())
import json, sys, urllib.request, difflib
src = open(sys.argv[1]).read()
tasks = [("common_sampler_sample", "common_sampler_draw"), ("common_sampler_free", "common_sampler_release")]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
tn = tms = td = ta = 0
for r in range(reps):
  for old, new in tasks:
    prompt = f"Rename `{old}` to `{new}` everywhere in this file. Output the complete updated file in one ```cpp block and nothing else.\n\n```cpp\n{src}```"
    body = json.dumps({"messages":[{"role":"user","content":prompt}],"max_tokens":3000,"seed":11+r,
                       "chat_template_kwargs":{"enable_thinking":False}}).encode()
    j = json.load(urllib.request.urlopen(urllib.request.Request(URL+"/v1/chat/completions", body, {"Content-Type":"application/json"}), timeout=900))
    t = j["timings"]; out = j["choices"][0]["message"]["content"]
    code = out.split("```cpp",1)[-1].split("```",1)[0].strip("\n")
    sim = difflib.SequenceMatcher(None, code, src.replace(old, new).strip("\n")).ratio()
    tn += t["predicted_n"]; tms += t["predicted_ms"]; td += t.get("draft_n",0); ta += t.get("draft_n_accepted",0)
    print(f"  {t['predicted_per_second']:6.1f} t/s  drafted {t.get('draft_n',0):5d} accepted {t.get('draft_n_accepted',0):5d}  n={t['predicted_n']}  match={sim:.4f}", flush=True)
print(f"COPY OVERALL {1000*tn/tms:.1f} t/s  accepted {ta}/{td} ({100*ta/max(1,td):.0f}%)  tok/step {tn/max(1,tn-ta):.2f}")
