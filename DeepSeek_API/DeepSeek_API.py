from openai import OpenAI
import os

def get_api_client():
    """获取DeepSeek API客户端"""
    with open(os.path.join('./', 'api_key.txt'), 'r') as f:
        key = f.readline().strip()
        return OpenAI(api_key=key, 
                      base_url="https://api.deepseek.com/v1")

def get_response_by_client(client, system_msg, prompt):
    try:
        response = client.chat.completions.create(
                model="deepseek-reasoner", # V3: deepseek-chat, R1:deepseek-reasoner
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                max_tokens=1500,
            )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        raise

client = get_api_client()
res = get_response_by_client(client=client, system_msg="请你作为一名作文教师", prompt="告诉我如何提高作文写作水平？")
print(res)
