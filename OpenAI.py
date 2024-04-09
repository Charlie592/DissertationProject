#OpenAI
from dotenv import load_dotenv
from openai import OpenAI
import os
import time
import json
import streamlit as st
import re 

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
          {"role": "system", "content": "Provide the following fields in JSON dict \"Cluster name\",\"heading\",\"description\",\"buzxwords\",\"key points\".  For each cluster provide a detailed, easily understandable description of at least 100 words and key points semicolon delimted (make sure there is always more than one keypoint for each cluster) plus buzzwords with a heading made with a maximum of 5 words, include text formating but do not assume its about any particular subject or to any particular audience:"},
          {"role": "user", "content": new_descriptions},
      ],
  )
  #print (response.choices[0].message.content)
  data = json.loads(response.choices[0].message.content)
  #print(data)
  combined_array = []

  # Loop through each cluster in the JSON data
  for cluster in data['clusters']:
      # Combine the heading and key points with a line feed
      key_points_list = cluster['key points'].split('; ')
      key_points_html = '<ul>' + ''.join(f'<li>{point}</li>' for point in key_points_list) + '</ul>'

      combined_text = f"""
      <div style='font-weight: bold;'>{cluster['heading']}</div>
      <div>{cluster['description']}</div>
      <div style='font-weight: bold;'>Buzzwords:</div>
      <div>{cluster['buzzwords']}</div>
      <div style='font-weight: bold;'>Key points:</div>
      {key_points_html}
      """

      # Add the combined text to the array
      combined_array.append(combined_text)

  return combined_array
