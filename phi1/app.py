import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


torch.set_default_device("cpu")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-1_5", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
inputs = tokenizer(
    "create a poem about a dog",
    return_tensors="pt",
    return_attention_mask=False,
)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
