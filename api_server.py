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
        # base_model_path="Qwen/Qwen2-7B-Instruct",
        # ea_model_path="yuhuili/EAGLE-Qwen2-7B-Instruct",
        base_model_path="meta-llama/Meta-Llama-3-8B",
        ea_model_path="yuhuili/EAGLE-LLaMA3-Instruct-8B",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=-1
        )
        self.model.eval()

    # def decode_request(self, request):
    #     # print(request)
    #     print("Received request:", request)
    #     return self.model.tokenizer.apply_chat_template(request.messages)

    # def predict(self, conversation, context):
    #         """
    #         conversation: OpenAI-style list of messages
    #         context: Contains parameters like `temperature`, `max_tokens`
    #         """
    #         max_gen_len = context.get("max_tokens", 128)
    #         temperature = context.get("temperature", 0.7)

    #         # 格式化输入
    #         formatted_conversation = self.model.tokenizer.apply_chat_template(
    #             conversation, add_generation_prompt=True
    #         )

    #         # Tokenize 输入
    #         input_ids = self.model.tokenizer([formatted_conversation]).input_ids
    #         input_ids = torch.as_tensor(input_ids).to(self.model.base_model.device)

    #         # 生成模型输出
    #         raw_output = self.model.eagenerate(input_ids, temperature=temperature, max_new_tokens=max_gen_len)
    #         print(raw_output)
    #         # 逐步返回 token 以支持流式响应
    #         for token in raw_output[0]:
    #             yield {"role": "assistant", "content": self.model.tokenizer.decode([token])}

    def predict(self, request):
        # 确保 request 是字典或列表
        print("reqs:",request)
        if isinstance(request, dict):
            messages = request.get("messages", [])
        elif isinstance(request, list):
            messages = request
        else:
            raise ValueError(f"Unexpected request format: {type(request)}")

        # 确保 messages 不为空
        if not messages:
            raise ValueError("Empty messages list received.")
        # print("meesages:", messages)
        # 取最后一条用户消息
        prompt = messages[-1]["content"]
        
        # Tokenize 输入
        input_ids = self.model.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.base_model.device)
        print("input_ids:",input_ids)
        # 生成模型输出 (支持流式返回)
        raw_output = self.model.eagenerate(input_ids, temperature=0.5, max_new_tokens=512)
        # print(raw_output)
        # 逐个 token 解码，避免一次性返回
        new_tokens = raw_output[0, input_ids.shape[1]:]
        print("new_token:",new_tokens)
        print(self.model.tokenizer.decode(new_tokens, skip_special_tokens=True))
        for token in new_tokens:
            yield self.model.tokenizer.decode(token, skip_special_tokens=True)
       
            



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