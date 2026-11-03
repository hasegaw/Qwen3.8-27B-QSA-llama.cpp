import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LLAMA_REPO", os.path.abspath(os.path.join(HERE, "../../..")))
URL = os.environ.get("QWEN38_URL", "http://127.0.0.1:18080")
TESTIMG = os.environ.get("QWEN38_TESTIMG", os.path.join(HERE, "testimg"))
OUT = os.environ.get("QWEN38_OUT", os.getcwd())
import json, urllib.request, collections, re, sys, html
U=URL
def post(path, obj):
    return json.load(urllib.request.urlopen(urllib.request.Request(U+path, json.dumps(obj).encode(), {"Content-Type":"application/json"}), timeout=900))
cnt = collections.Counter({int(k): v for k, v in json.load(open(os.path.join(OUT, "tokcounts.json")))})
add = collections.Counter()
def tok(text, w=1):
    for i in range(0, len(text), 100000):
        for t in post("/tokenize", {"content": text[i:i+100000]})["tokens"]: add[t] += w
prompts = ["東京の観光名所を10個紹介して、それぞれの見どころを説明してください。", "機械学習とディープラーニングの違いを初心者向けに説明してください。",
 "日本の歴史を、縄文時代から現代まで簡潔にまとめてください。", "健康的な食生活のためのアドバイスを詳しく教えてください。",
 "Pythonでクイックソートを実装し、コードの各部分を日本語で説明してください。", "敬語の使い方について、具体例を交えて説明してください。",
 "日本の会社で働く際のビジネスマナーを説明してください。", "地球温暖化の原因と対策について論じてください。", "短い推理小説を書いてください。",
 "量子コンピュータの仕組みをわかりやすく説明してください。", "请详细介绍中国的四大发明及其影响。", "用中文解释什么是区块链技术。",
 "写一篇关于春节习俗的文章。", "请比较Python和Java的优缺点。", "한국의 전통 음식에 대해 설명해 주세요.", "인공지능의 미래에 대해 써 주세요.",
 "Translate into Japanese: The quick development of renewable energy is changing how countries plan their power grids, and storage remains the main challenge.",
 "Translate into Chinese: Software engineers should write tests before refactoring code, because tests protect behavior during changes.",
 "以下の英文を日本語に訳してください: Large language models generate text one token at a time, and speculative decoding lets a small draft propose several tokens that the large model verifies in one pass.",
 "日本語で、Dockerの基本的な使い方を手順ごとに説明してください。"]
for p in prompts:
    r = post("/v1/chat/completions", {"messages":[{"role":"user","content":p}],"max_tokens":700,"chat_template_kwargs":{"enable_thinking":False}})
    tok(r["choices"][0]["message"]["content"], 3)
print("gen done", flush=True)
for f in [os.environ.get("JA_README", "README.ja.md"), os.environ.get("JA_HTML", "index.html")]:
    t = open(f, errors="ignore").read()
    if f.endswith(".html"): t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    tok(t, 2)
new = [t for t in add if t not in cnt]
print("tokens in new corpus:", len(add), "of which not in old corpus:", len(new))
total = cnt + add
N = 65536; ids = []; seen = set()
def put(i):
    if i not in seen and len(ids) < N: seen.add(i); ids.append(i)
for i in range(248000, 248320): put(i)
for t, _ in total.most_common(): put(t)
i = 0
while len(ids) < N: put(i); i += 1
open(os.path.join(OUT, "draftvocab-64k-ml.txt"), "w").write("# top tokens (English, code, Japanese, Chinese, Korean) + specials\n" + "\n".join(map(str, sorted(ids))) + "\n")
old = set(int(x) for x in open(os.path.join(OUT, "draftvocab-64k.txt")).read().split() if x.isdigit())
print("new list", len(ids), "| changed vs old:", len(set(ids) - old))
