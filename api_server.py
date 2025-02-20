import os

import litserve as ls

# from m2d.config import GenConfig
# from m2d.formatter import Formatter
# from m2d.model.llama import M2DLlama

# GEN_CONFIG = GenConfig.from_path(
#     os.environ.get("GEN_CONFIG_PATH", None)
# )

# FORMATTER = Formatter.from_path(
#     os.environ.get("FORMAT_CONFIG_PATH", None)
# )

from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
class EALitAPI(ls.LitAPI):
    def setup(self, device):
        print(f"Using device={device}")
        self.model= EaModel.from_pretrained(
        base_model_path="Qwen/Qwen2-7B-Instruct",
        ea_model_path="yuhuili/EAGLE-Qwen2-7B-Instruct",
        # base_model_path="meta-llama/Meta-Llama-3-8B",
        # ea_model_path="yuhuili/EAGLE-LLaMA3-Instruct-8B",
        # torch_dtype=torch.float16,
        torch_dtype=torch.bfloat16, #bf16 for qwen 2
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=-1
        )
        self.model.eval()

    def predict(self, conversation):
        """ 处理对话，返回流式响应 """

        
        print("conversation:",conversation)
        # **2️⃣ 过滤非必要字段，确保 messages 只有 "role" 和 "content"**
        filtered_conversation = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in conversation
        ]

        # **3️⃣ 取最后一条 "user" 消息作为 prompt**
        prompt = None
        for msg in reversed(filtered_conversation):
            if msg["role"] == "user":
                prompt = msg["content"]
                break

        if prompt is None:
            raise ValueError("No valid user message found in conversation.")

        # print("Prompt:",prompt)
        # **4️⃣ Tokenize 输入**
        input_ids = self.model.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.base_model.device)

        # **5️⃣ 生成模型输出**
        raw_output = self.model.eagenerate(input_ids, temperature=0.5, max_new_tokens=512)

        # **6️⃣ 去除输入部分，确保只返回新生成的 tokens**
        new_tokens = raw_output[0, input_ids.shape[1]:]

        # **7️⃣ 逐步解码并返回**
        output_text = self.model.tokenizer.decode(new_tokens)
        print("output text",output_text)
        print("----------------------------------------")
        # yield from output_text
        for token in new_tokens:
            yield self.model.tokenizer.decode(token)

    # def predict(self, request):
    #     # 确保 request 是字典或列表
    #     print("reqs:",request)
    #     if isinstance(request, dict):
    #         messages = request.get("messages", [])
    #     elif isinstance(request, list):
    #         messages = request
    #     else:
    #         raise ValueError(f"Unexpected request format: {type(request)}")

    #     # 确保 messages 不为空
    #     if not messages:
    #         raise ValueError("Empty messages list received.")
    #     # print("meesages:", messages)
    #     # 取最后一条用户消息
    #     prompt = messages[-1]["content"]
        
    #     # Tokenize 输入
    #     input_ids = self.model.tokenizer(prompt, return_tensors="pt").input_ids
    #     input_ids = input_ids.to(self.model.base_model.device)
    #     print("input_ids:",input_ids)
    #     # 生成模型输出 (支持流式返回)
    #     raw_output = self.model.eagenerate(input_ids, temperature=0.5, max_new_tokens=512)
    #     # print(raw_output)
    #     # 逐个 token 解码，避免一次性返回
    #     new_tokens = raw_output[0, input_ids.shape[1]:]
    #     print("new_token:",new_tokens)
    #     print(self.model.tokenizer.decode(new_tokens, skip_special_tokens=True))
    #     for token in new_tokens:
    #         yield self.model.tokenizer.decode(token, skip_special_tokens=True)
       
            



    # def predict(self, prompt):
    #     for chunk in "This is a sample generated output".split():
    #             yield chunk
    #             # This
    #             # is
    #             # a
    #             # sample
    #             # generated
    #             # output


if __name__ == "__main__":
    api = EALitAPI()
    server = ls.LitServer(
        api, 
        devices=1, 
        spec=ls.OpenAISpec()
    )
    server.run(port=8000)