#OpenAI
from dotenv import load_dotenv
from openai import OpenAI
import os
import time
import json
import streamlit as st

load_dotenv()

def generate_summary(descriptions):
  client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
  )
  new_descriptions = ', '.join(descriptions)
  response = client.chat.completions.create(
      model="gpt-3.5-turbo-0125",
      max_tokens=4096,
      temperature=0.7,
      messages=[
          {"role": "system", "content": "Provide the following fields in JSON dict \"Cluster name\",\"heading\",\"description\",\"buzxwords\",\"key points\".  For each cluster provide a detailed, easily understandable description of at least 100 words and key points plus buzzwords with a heading made with a maximum of 5 words, include text formating but do not assume its about any particular subject or to any particular audience:"},
          {"role": "user", "content": new_descriptions},
      ],
  )
  print (response.choices[0].message.content)
  data = json.loads(response.choices[0].message.content)
  print(data)
  combined_array = []

  # Loop through each cluster in the JSON data
  for cluster in data['clusters']:
      # Combine the heading and key points with a line feed
      combined_text = f"***{cluster['heading']}***\n**{cluster['description']}**\n**Buzzwords**\n{cluster['buzzwords']}\n**Key points**\n{cluster['key points']}"
      # Add the combined text to the array
      combined_array.append(combined_text)

  return combined_array
