from openai import OpenAI
import time

# 设置你的API密钥
API_KEY ="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"

# 初始化客户端
client = OpenAI(api_key=API_KEY)

def test_gpt(prompt, model="gpt-4", max_tokens=100):
    """简单的GPT API测试函数"""
    try:
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        
        elapsed_time = time.time() - start_time
        answer = response.choices[0].message.content
        
        print("测试成功！")
        print(f"响应时间: {elapsed_time:.2f}秒")
        print(f"回答内容: {answer}")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")

# 运行测试
test_gpt("用中文简单介绍一下你自己")