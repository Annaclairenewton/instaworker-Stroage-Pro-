import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指向你解压后的路径
model_path = r"C:\Users\16693\OneDrive\Desktop\instaworker-Stroage-Pro-\final_gemma_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32, # CPU 运行 float32 最稳
    device_map="cpu"
)

# 1. 手动构造最原始的 Prompt，不要用 apply_chat_template
# 这样能避开 tokenizer 自动添加的干扰标签
raw_query = "Should I place an order for Panel Board 100A?"
prompt = f"<start_of_turn>user\n{raw_query}<end_of_turn>\n<start_of_turn>model\n"

inputs = tokenizer(prompt, return_tensors="pt")

# 2. 严格限制生成行为
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,      # 限制长度，防止它开始复读整个数据集
        do_sample=False,        # 设为 False 保证输出稳定
        repetition_penalty=1.5, # 强力惩罚重复，阻止它不停输出 <escape>
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bad_words_ids=[[tokenizer.encode("<escape>", add_special_tokens=False)[0]]]
    )

# 3. 清理输出逻辑
full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

# 只取 model 之后的内容
if "model\n" in full_output:
    response = full_output.split("model\n")[-1]
else:
    response = full_output

# 4. 暴力过滤掉那些无意义的标签
clean_response = response.replace("<unused85>", "").replace("<escape>", "").split("<end_of_turn>")[0]

print("\n" + "="*40)
print("模型最终回答：")
print(clean_response.strip())
print("="*40)