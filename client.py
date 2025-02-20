import requests

# OpenAI API standard endpoint
SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"

request_data = {
    "model": "my-gpt2",
    "stream": True,  # This must be True for streaming response
    "messages": [
        {"role": "user", "content": "How can I help you today?"}
    ]
}

if __name__ == "__main__":
    response = requests.post(SERVER_URL, json=request_data, stream=True)
    for chunk in response.iter_lines():
        print(chunk.decode("utf-8"))
