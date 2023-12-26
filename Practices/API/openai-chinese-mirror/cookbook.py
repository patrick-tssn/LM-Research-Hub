import openai
from os import getenv

from dotenv import load_dotenv
load_dotenv()

# mirror 查询余量：https://api.chatanywhere.cn/
OPENAI_TOKEN = getenv("OPENAI_TOKEN")
openai.api_key = OPENAI_TOKEN
openai.api_base = "https://api.chatanywhere.com.cn/v1"

# # close AI 查询余量：https://console.closeai-asia.com/account/usage
# CLOSEAI_TOKEN = getenv("CLOSEAI_TOKEN")
# openai.api_key = CLOSEAI_TOKEN
# openai.api_base = "https://api.closeai-proxy.xyz/v1"

# # original WICT
# ORI_OPENAI_TOKEN = getenv("ORI_OPENAI_TOKEN")
# openai.api_key = ORI_OPENAI_TOKEN

# prompt guidance: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/completions

promt = """

This is a tweet sentiment classifier

Tweet: "I loved the new Batman movie!"
Sentiment: Positive

Tweet: "I hate it when my phone battery dies." 
Sentiment: Negative

Tweet: "My day has been 👍"
Sentiment: Positive

Tweet: "This is the link to the article"
Sentiment: Neutral

Tweet: "This new music video blew my mind it is noisy"
Sentiment:

"""

# # support models: ada, babbage, curie, text-davinci
response = openai.Completion.create(
    model="text-davinci-003",
    prompt=promt,
    temperature=0.7,
    max_tokens=512,
    top_p=0.5,
    frequency_penalty=0,
    presence_penalty=0,
    n=1,
    logprobs=1
)
response_text = response.choices[0]['text'].lstrip('\n').rstrip('\n')
print(response_text)
print(response)
        
# # support models: gpt-3.5-turbo, gpt-4
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": promt}
    ],
    temperature=0.7,
    max_tokens=512,
    top_p=0.5,
    frequency_penalty=0,
    presence_penalty=0,
    n=1,
    # logprobs=1
)
response_text = response.choices[0].message['content'].lstrip('\n').rstrip('\n')
print(response_text)
print(response)