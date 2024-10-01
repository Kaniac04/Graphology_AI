import os
from mistralai import Mistral
from constants import Mistral_apikey

def get_traits_from_AI(features : dict):
    SYSTEM_PROMPT=""
    with open(r"C:\Users\User\Documents\GitHub\AI_Graphology\Graphology_AI\context.md", 'r') as prompt_file : 
        SYSTEM_PROMPT = prompt_file.read()

    api_key = Mistral_apikey
    model = "mistral-large-latest"

    client = Mistral(api_key=api_key)
    messages = [{
            "role": "system",
            "content": f"{SYSTEM_PROMPT}",
        },
        {
            "role": "user",
            "content": f"The available features for user handwriting are : {str(features)}",
        }
    ]
    chat_response = client.chat.complete(
        model = model,
        messages = messages,
        
    )

    return(chat_response.choices[0].message.content)

