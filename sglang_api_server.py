import os

import litserve as ls



import sglang as sgl
# from fastchat.model import get_conversation_template
import torch
class EALitAPI(ls.LitAPI):
    def setup(self, device):
        print(f"Using device={device}")
        self.model= sgl.Engine(model_path="Qwen/Qwen2-7B-Instruct")

    def predict(self, conversation):
        """ 处理对话，返回流式响应 """

        
        print("conversation:",conversation)
        # **2️⃣ 过滤非必要字段，确保 messages 只有 "role" 和 "content"**
        filtered_conversation = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in conversation
        ]

        input_ids = self.model.tokenizer_manager.tokenizer.apply_chat_template(
        filtered_conversation, 
        add_generation_prompt=True, 
        return_tensors='pt', 
        return_dict=True,
        # tokenize = False,)
        )["input_ids"].to(self.model.base_model.device)
        print(input_ids)

        
        # # **4️⃣ Tokenize 输入**
        # input_ids = self.model.tokenizer(prompt, return_tensors="pt").input_ids
        # input_ids = input_ids.to(self.model.base_model.device)

        # **5️⃣ 生成模型输出**
        sampling_params = {
                "temperature":0,
                "max_new_tokens": 512,
            }
        raw_output = self.model.generate(input_ids=input_ids, sampling_params=sampling_params)

        # **6️⃣ 去除输入部分，确保只返回新生成的 tokens**
        new_tokens = raw_output[0, input_ids.shape[1]:]

        # **7️⃣ 逐步解码并返回**
        output_text = self.model.tokenizer_manager.tokenizer.decode(new_tokens)
        print("output text",output_text)
        print("----------------------------------------")
        # yield from output_text
        # for token in new_tokens:
        #     yield self.model.tokenizer.decode(token)
        yield output_text



if __name__ == "__main__":
    api = EALitAPI()
    server = ls.LitServer(
        api, 
        devices=1, 
        spec=ls.OpenAISpec()
    )
    server.run(port=8000)