import base64
import requests

with open("data/golden/images/5174.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")

# ---- 開關區 ----
USE_DETAIL    = True   # 測試 1 切到 False
USE_STREAM    = False    # 測試 2 用 True
USE_SYSTEM    = True   # 測試 3 用 True 看會不會壞
# ---------------

img_obj = {"url": f"data:image/jpeg;base64,{b64}"}
if USE_DETAIL:
    img_obj["detail"] = "high"

messages = []
if USE_SYSTEM:
    messages.append({"role": "system", "content": "你是一個專業的問答助手。"})
messages.append({
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": img_obj},
        {"type": "text", "text": "請用一句話描述你看到的圖片內容。"},
    ],
})

payload = {
    "model": "gpt-4o",
    "messages": messages,
    "stream": USE_STREAM,
    "temperature": 0.8,
    "top_k": 40, "top_p": 0.95, "min_p": 0.05,
    "max_tokens": -1,
}

r = requests.post(
    "https://dgx-02-8081.wade0426.me/v1/chat/completions",
    json=payload, headers={"Authorization": "Bearer test"},
    stream=USE_STREAM, timeout=600,
)
if USE_STREAM:
    for line in r.iter_lines():
        if line: print(line.decode("utf-8", errors="replace"))
else:
    print(r.status_code, r.text[:2000])