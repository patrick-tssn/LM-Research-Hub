import json
import openai
from os import getenv

from dotenv import load_dotenv
# openai.log = "debug"
load_dotenv()

# chat anywhere
OPENAI_TOKEN = getenv("OPENAI_TOKEN")
openai.api_key = OPENAI_TOKEN
openai.api_base = "https://api.chatanywhere.com.cn/v1"

# # close AI
# CLOSEAI_TOKEN = getenv("CLOSEAI_TOKEN")
# openai.api_key = CLOSEAI_TOKEN
# openai.api_base = "https://api.closeai-proxy.xyz"



# 非流式响应
# completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world!"}])
# print(completion.choices[0].message.content)

def gpt_35_api_stream(messages: list):
    """为提供的对话消息创建新的回答 (流式传输)

    Args:
        messages (list): 完整的对话消息
        api_key (str): OpenAI API 密钥

    Returns:
        tuple: (results, error_desc)
    """
    try:
        response = openai.ChatCompletion.create(
            model='gpt-4', # gpt-3.5-turbo
            messages=messages,
            stream=True,
        )
        completion = {'role': '', 'content': ''}
        for event in response:
            if event['choices'][0]['finish_reason'] == 'stop':
                # print(f'收到的完成数据: {completion}')
                break
            for delta_k, delta_v in event['choices'][0]['delta'].items():
                # print(f'流响应数据: {delta_k} = {delta_v}')
                completion[delta_k] += delta_v
        messages.append(completion)  # 直接在传入参数 messages 中追加消息
        return (True, '')
    except Exception as err:
        return (False, f'OpenAI API 异常: {err}')
    

if __name__ == '__main__':
    
    # while True:
    #     input_text = input('You: ').strip()
    #     if len(input_text) == 0:
    #         print('**no response**')
    #         continue
    #     else:
    #         messages = [{'role': 'user','content': input_text},]
    #         print(gpt_35_api_stream(messages))
    #         print('ChatGPT: ' + messages[1]['content'])
    
    prompt = '润色这封情况说明邮件：各位领导好， 本人汪宇轩，特此邮件说明企业微信打卡的迟到早退情况。因为企业微信打卡没有后台自动打卡功能（其自动打卡功能需要手动打开APP），出勤后需要主动打开手机APP进行打卡。本人出勤后没有养成打开手机的习惯，只在小憩时想起打卡，故时有迟到。下班签退因为怕忘记签退，会在晚饭时间主动多次打卡，由于晚饭时间早于签退时间，故时有早退。至于实际出勤情况，可参考同院无线网登陆日志（本人使用个人笔记本 mac pro 2020 ip地址 10.1.121.116），如有需求，我可寻求科技大厦物业调取7月出入监控或者请4楼保洁阿姨做旁证。本人生活单调，包括节假日和周末，大部分时间都在通院度过，虽然能力一般，可保证出勤时间。之后，我会设置多个闹钟提醒自己打卡，以免造成不必要的麻烦。'
    messages = [{'role': 'user','content': prompt},]
    print(gpt_35_api_stream(messages))
    # print(messages)
    print(messages[1]["content"])
