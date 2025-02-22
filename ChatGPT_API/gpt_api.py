
from openai import OpenAI
import os


def get_api_client():
    # 获取 OpenAI access key
    with open(os.path.join('./', 'api_key.txt'), 'r') as f:
        key = f.readline().strip()
        client = OpenAI(api_key=key)
        return client


def get_response_by_client(client, model, prompt, max_tokens, temperature, logprobs):
    response = client.completions.create(
    model=model,
    prompt=prompt,
    max_tokens=max_tokens,  # 设置生成的最大令牌数
    temperature=temperature,  # 控制输出的随机性，0-1之间的值，值越高输出越随机
    logprobs=logprobs, # log-probabilities 是模型生成每个 token 时的概率，可以用来分析模型对生成的每个 token 的信心度。
    stop=["END"],
    )
    return response

# 创建请求的提示文本
prompt = """
请帮我解释下什么是机器学习算法？机器学习包括哪些算法模型？机器学习相比统计学习有什么优势？？
"""

client = get_api_client()
model = "gpt-3.5-turbo-instruct"
assert model in [
    'gpt-4o-realtime-preview', 'dall-e-2', 'text-embedding-ada-002',
    'gpt-4o-realtime-preview-2024-10-01', 'gpt-4-1106-preview',
    'text-embedding-3-large', 'babbage-002', 'gpt-4o-2024-11-20',
    'o1-mini', 'davinci-002', 'o1-mini-2024-09-12', 'whisper-1',
    'dall-e-3', 'o1-preview', 'gpt-3.5-turbo-16k', 'o1-preview-2024-09-12',
    'gpt-4-0125-preview', 'gpt-4-turbo-preview', 'omni-moderation-latest',
    'omni-moderation-2024-09-26', 'tts-1-hd-1106', 'gpt-4',
    'gpt-4-0613', 'gpt-4o-mini', 'gpt-4o-mini-2024-07-18',
    'gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 'text-embedding-3-small',
    'gpt-4-turbo', 'tts-1-hd', 'gpt-4o', 'gpt-4o-2024-08-06',
    'gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-instruct',
    'gpt-4o-audio-preview', 'gpt-4o-audio-preview-2024-10-01', 'tts-1',
    'tts-1-1106', 'gpt-3.5-turbo-instruct-0914', 'chatgpt-4o-latest',
    'gpt-4o-2024-05-13'
]
response = get_response_by_client(client, model, prompt, max_tokens=1000, temperature=0.0, logprobs=10)

# 打印生成的文本
print(response)
print(response.choices[0].text.strip())
