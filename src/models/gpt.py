import os

from openai import OpenAI

client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
  base_url=os.environ.get("OPENAI_BASE_URL", "https://api.pumpkinaigc.online/v1"),
  timeout=60,
)

def call_gpt(
    user_prompt, 
    system_prompt='你是个语言能力和逻辑理解能力很强的AI助手',
    model='gpt-5',
    print_mode=False,
):
  messages = [{'role': 'system', 'content': system_prompt},
              {'role': 'user', 'content': user_prompt}]
  try:
    stream = client.chat.completions.create(
      model=model,
      messages=messages,
      stream=True,
      timeout=300,
      reasoning_effort="high",
    )
    text = ''
    for chunk in stream:
      if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        text += content
        if print_mode:
            print(content, end="")
    if print_mode:
        print('')
    return text
  except Exception as e:
    return str(e)

