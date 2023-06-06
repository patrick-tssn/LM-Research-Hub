import json
import requests
import argparse



if __name__ == '__main__':
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        }
    reset_url = "http://127.0.0.1:8088/claude/reset"
    chat_url = "http://127.0.0.1:8088/claude/chat"
    response = []
    while True:
        input_text = input('You: ').strip()
        if len(input_text) == 0:
            print('**no response**')
            continue
        if input_text == "\\reset":
            response = requests.post(reset_url, headers=headers)
            assert response.status_code == 200
            print(response.text)
        else:
            data = {'prompt': input_text}
            response = requests.post(chat_url, headers=headers, json=data)
            assert response.status_code == 200
            response = eval(response.text)["claude"]
            print(f"Claude: {response}")
        

        