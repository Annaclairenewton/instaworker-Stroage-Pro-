import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# 必须加 CORS，否则前端连不上！
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 1. 路径检查 (确保 r 后面是你解压的准确路径)
MODEL_PATH = r"C:\Users\16693\OneDrive\Desktop\instaworker-Stroage-Pro-\final_gemma_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, device_map="cpu")

class ChatReq(BaseModel):
    message: str # 确保这个键名和前端发送的一致

@app.post("/api/chat")
async def chat(req: ChatReq):
    # 构造 Prompt
    prompt = f"<start_of_turn>user\n{req.message}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        # 限制长度，防止生成太久
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False, repetition_penalty=1.5)
    
    # 拿到原始文本
    full_txt = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    
    # --- 暴力清洗：只要有 call: 就显示调用中，否则显示文本 ---
    if "call:" in full_txt:
        res = "🔧 系统正在调用库存数据库检索中..."
    else:
        # 过滤掉标签，只留纯文字
        res = full_txt.replace("<unused85>", "").split("<")[0].strip()
        if not res: res = "我正在为您查询相关信息..."
        
    return {"reply": res}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)