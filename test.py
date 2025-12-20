from openai import OpenAI
import os
print('OPENAI_API_KEY =', bool(os.environ.get('OPENAI_API_KEY')))
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
resp = client.responses.create(model='gpt-5-mini', input='테스트 버튼 OK')
print('응답:', resp.output_text)
