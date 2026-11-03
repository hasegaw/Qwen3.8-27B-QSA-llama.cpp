import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LLAMA_REPO", os.path.abspath(os.path.join(HERE, "../../..")))
URL = os.environ.get("QWEN38_URL", "http://127.0.0.1:18080")
TESTIMG = os.environ.get("QWEN38_TESTIMG", os.path.join(HERE, "testimg"))
OUT = os.environ.get("QWEN38_OUT", os.getcwd())
import json, glob, collections, urllib.request, sys, os
U=URL
def post(path, obj):
    return json.load(urllib.request.urlopen(urllib.request.Request(U+path, json.dumps(obj).encode(), {"Content-Type":"application/json"}), timeout=900))
cnt = collections.Counter()
def add(text):
    for i in range(0, len(text), 200000):
        cnt.update(post("/tokenize", {"content": text[i:i+200000]})["tokens"])
prompts = ["Explain how a refrigerator works.", "Write a haiku sequence about autumn.", "What are the main causes of inflation?",
 "Write a Python script that parses a CSV file and computes summary statistics.", "Explain recursion to a 10 year old.",
 "Write a JavaScript debounce function and explain it.", "Summarize the plot of a fictional mystery novel you invent.",
 "What is the difference between TCP and UDP? Use a table.", "Write a SQL query to find the top 5 customers by revenue, then explain it.",
 "Give me a weekly meal plan for a vegetarian.", "Explain the theory of evolution.", "Write a bash script to back up a directory with rotation.",
 "日本の四季について説明してください。", "请解释一下机器学习的基本概念。", "Explique la photosynthèse en détail.", "Erkläre die Relativitätstheorie einfach.",
 "Explica la historia del imperio romano brevemente.", "Write a cover letter for a software engineering job.",
 "Write a C++ class implementing a binary search tree.", "Prove that the square root of 2 is irrational.",
 "Solve: a train leaves at 3pm going 60 mph, another at 4pm going 80 mph; when does the second catch up? Show steps.",
 "Write a Go HTTP server with JSON endpoints.", "Explain how vaccines train the immune system.", "Write a product description for a smart water bottle.",
 "Explain Kubernetes pods, services and deployments.", "Write a dialogue between a detective and a suspect.", "What are good habits for mental health?",
 "Explain quantum entanglement with an analogy.", "Write a TypeScript React component for a todo list.", "Describe the water cycle.",
 "Write an email asking a manager for a deadline extension.", "Explain gradient descent with math.", "Write a Java program for a bank account with threads.",
 "List pros and cons of remote work.", "Explain how blockchains reach consensus.", "Write a limerick about a cat and a laser pointer.",
 "Explain the French Revolution's causes.", "Write a Dockerfile for a Python Flask app and explain each line.", "Explain big-O notation with examples.",
 "Write a markdown README for a CLI tool called 'tidy'."]
for p in prompts:
    r = post("/v1/chat/completions", {"messages":[{"role":"user","content":p}],"max_tokens":700,"chat_template_kwargs":{"enable_thinking":False}})
    c = r["choices"][0]["message"]["content"]
    for _ in range(3): add(c)   # weight model output higher
print("gen done", len(cnt), flush=True)
files = glob.glob("/usr/lib/python3*/**/*.py", recursive=True)[:1500] + glob.glob(sys.argv[1]+"/**/*.md", recursive=True) \
      + glob.glob(sys.argv[1]+"/src/**/*.cpp", recursive=True)[:80] + glob.glob("/usr/share/doc/**/*.txt", recursive=True)[:300]
buf = []
for f in files:
    try: buf.append(open(f, errors="ignore").read())
    except Exception: pass
add("".join(buf)[:40_000_000])
print("corpus done", len(cnt), sum(cnt.values()), flush=True)
json.dump(cnt.most_common(), open(sys.argv[2], "w"))
