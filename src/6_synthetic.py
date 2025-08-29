"""
Generate synthetic comment data
"""

import json
import requests

TOPIC_SEED = "action movies"
START_THEME = "mission impossible, terminator, classic"
END_THEME = "marvel universe"

GENERATE_COMMENT_PROMPT = (
    "You are roleplaying as a Reddit Comment Generator Machine, "
    "that can generate comments that sound like something "
    "you might find on Reddit.  You can imitate a variety of user styles and "
    "exhibit a range of sentiment.\n\n"
    "Your task today is to perform generation on the topic of {topic}. " 
    "I need you to generate {num_comments} unique comments on this topic, "
    "focusing on the following specific theme(s): {themes}.\n\n "
    "Provide just the {num_comments} comments, nothing else. "
    "Each comment can be 1-3 sentences, "
    "and separate each one by three asterisks, like this: '***'."
    ""
)

try:
    # response = requests.get("http://ollama:80/api/tags")
    # response.raise_for_status()
    # models = response.json()
    # print("Models:", models)

    payload = {
        "model": "llama3.2:latest",
        "messages": [{
            "role": "user", 
            "content": GENERATE_COMMENT_PROMPT.format(
                topic=TOPIC_SEED,
                num_comments=3,
                themes=START_THEME,
            )
        }],
        "stream": False
    }
    response = requests.post("http://ollama:80/api/chat", json=payload)
    response.raise_for_status()
    
    data = json.loads(response.text.strip())
    message = data['message']['content']
    print(message)

    

except requests.RequestException as e:
    print("Error sending chat request:", e)