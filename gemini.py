from google import genai

client = genai.Client(api_key="AIzaSyA57TVnLOQ0VcQoZq467EMX7ypEfvNNPao")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a few words"
)
print(response.text)