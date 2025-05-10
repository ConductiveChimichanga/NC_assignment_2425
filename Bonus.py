import openai

# Gets access to the LLM using an api key
client = openai.OpenAI(api_key="Your-API-Key")
# Dictionary containing 10 dishes and the users opinion about them
predicted_class = {
    "pizza": "likes",
    "sushi": "dislikes",
    "burger": "neutral",
    "salad": "likes",
    "pasta": "likes",
    "ice cream": "likes",
    "escargot": "dislikes",
    "chicken wings": "likes",
    "fish and chips": "neutral",
    "pork chop": "likes"
}

# Creates the prompt
input_text = f"User food preferences: {predicted_class}. Write a description of the food preference of the user based on this dictionary."

# Defines the role the LLM should have, sends the prompt to the LLM and puts the reaction in response
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_text}
    ]
)

# Prints the response of the LLM
print(response.choices[0].message.content)
