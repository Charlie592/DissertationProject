#OpenAI
from dotenv import load_dotenv
from openai import OpenAI
import os
import time
import json
import streamlit as st
import re 

load_dotenv()

def generate_summary(descriptions, retry_lmit=3):
  client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
  )
  new_descriptions = ', '.join(descriptions)

  attempts = 0
  while attempts < retry_lmit:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        max_tokens=4096,
        temperature=0.7,
        messages=[
            {"role": "system", "content": "Your task is to generate a JSON structure containing clusters of data. Each cluster should include details such as 'Cluster name', 'heading', 'description', 'buzzwords', and 'key points'. The JSON structure must start with a 'clusters' key, which holds an array of cluster objects. Each object in the array represents a cluster and should provide a detailed, easily understandable description of at least 100 words and key points semicolon delimited (make sure there is always more than one key point for each cluster) plus buzzwords with a heading made with a maximum of 5 words. Include text formatting but do not assume it's about any particular subject or to any particular audience. Ensure the response adheres strictly to this JSON format."},
            {"role": "user", "content": new_descriptions},
        ],
    )

    print(response.choices[0].message.content)

    try:
        data = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        attempts += 1
        time.sleep(20)  # Wait a bit before retrying
        continue

    combined_array = []
    if isinstance(data.get('clusters'), list):
        for cluster in data['clusters']:
            heading = cluster.get('heading', 'No Heading')
            description = cluster.get('description', 'No Description')
            buzzwords = cluster.get('buzzwords', 'No Buzzwords')
            key_points_str = cluster.get('key points', '')
            key_points_list = key_points_str.split('; ')

            key_points_html = '<ul>' + ''.join(f'<li>{point}</li>' for point in key_points_list) + '</ul>'

            combined_text = f"""
            <div style='font-weight: bold;'>{heading}</div>
            <div>{description}</div>
            <div style='font-weight: bold;'>Buzzwords:</div>
            <div>{buzzwords}</div>
            <div style='font-weight: bold;'>Key points:</div>
            {key_points_html}
            """

            combined_array.append(combined_text)
        
        return combined_array  # Successful processing, return the result
    else:
        print("'clusters' key is missing or not in expected format. Retrying...")
        attempts += 1
        time.sleep(20)  # Wait a bit before retrying



