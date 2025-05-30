import google.generativeai as genai

# Set your Gemini API key
genai.configure(api_key="**************************")

# Load the Gemini 2.0 Flash model
model = genai.GenerativeModel('gemini-2.0-flash')

# Prompt
response = model.generate_content("Write a story about a magic backpack.")

# Print the response
print(response.text)
