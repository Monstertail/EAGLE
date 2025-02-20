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

    def predict(self, prompt):
                # Tokenize 输入
        input_ids = self.model.tokenizer([prompt]).input_ids
        input_ids = torch.as_tensor(input_ids).to(self.model.base_model.device)

        # 生成模型输出
        raw_output = self.model.eagenerate(input_ids, temperature=0.5, max_new_tokens=512)
        print(raw_output)
        # 解码输出
        output_text = self.model.tokenizer.decode(raw_output[0])  # 取第一个序列

        # 逐个字符流式返回
        for token in output_text:
            yield token  # 逐字符返回，适用于流式生成


if __name__ == "__main__":
    api = EALitAPI()
    server = ls.LitServer(
        api, 
        devices=1, 
        spec=ls.OpenAISpec()
    )
    server.run(port=8000)