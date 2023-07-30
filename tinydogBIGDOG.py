import numpy as np
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from gpt4all import GPT4All, Embed4All
import openai
import os
import dotenv
import pandas as pd
import pickle

# You'll need to set your own .env file
dotenv.load_dotenv()

class GPT4AllChatbot:
    def __init__(self):
        self.model = GPT4All(model_name='yr_local_model.bin')
        self.embedder = Embed4All()

    def generate_response(self, query):
        with self.model.chat_session():
            return self.model.generate(prompt=query, top_k=1)

    def embed(self, text):
        return self.embedder.embed(text)


class OpenAIChatbot:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def generate_response(self, query):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
        )
        return response['choices'][0]['message']['content']


# Initialize the chatbots
gpt4all_chatbot = GPT4AllChatbot()
openai_chatbot = OpenAIChatbot()

# Set the threshold for the semantic search
semantic_search_threshold = 0.3

# Set the threshold for the cosine similarity score between the query and the chatbot's response
response_similarity_threshold = 0.6

# Initialize the chat history and embeddings
chat_history = []

# Check if the embeddings pickle file exists
if os.path.exists('embeddings.pkl'):
    # If it exists, load the existing embeddings
    with open('embeddings.pkl', 'rb') as f:
        chat_history_embeddings = pickle.load(f)
else:
    # If it doesn't exist, create a new list
    chat_history_embeddings = []

def semantic_search(query_embedding, history_embeddings):
    # Compute cosine similarity between the query embedding and each history embedding
    similarities = cosine_similarity(query_embedding, history_embeddings)
    # Get the index of the most similar history embedding
    most_similar_index = np.argmax(similarities)
    return chat_history[most_similar_index]

# Check if the JSON file exists
if os.path.exists('conversation.json'):
    # If it exists, load the existing conversations
    conversation_df = pd.read_json('conversation.json', orient='records', lines=True)
else:
    # If it doesn't exist, create a new DataFrame
    conversation_df = pd.DataFrame(columns=['User_Input', 'Bot_Response', 'User_Sentiment', 'Bot_Sentiment', 'User_Keywords', 'Bot_Keywords', 'Model'])

last_response = ""

def get_response(query):
    global last_response
    if last_response:
        query = last_response + " " + query

    query_embedding = np.array(gpt4all_chatbot.embed(query)).reshape(1, -1)

    if chat_history:
        similar_response = semantic_search(query_embedding, np.array(chat_history_embeddings))
        if similar_response >= semantic_search_threshold:
            query += ' ' + similar_response

    gpt4all_response = gpt4all_chatbot.generate_response(query)
    gpt4all_response_embedding = np.array(gpt4all_chatbot.embed(gpt4all_response)).reshape(1, -1)

    similarity_score_gpt4all = cosine_similarity(query_embedding, gpt4all_response_embedding)

    # Initialize model_used
    model_used = None

    if similarity_score_gpt4all >= response_similarity_threshold:
        response = gpt4all_response
        model_used = "smalldog"  # GPT-4All model
    else:
        openai_response = openai_chatbot.generate_response(query)
        openai_response_embedding = np.array(gpt4all_chatbot.embed(openai_response)).reshape(1, -1)
        response = openai_response
        model_used = "BIGDOG"  # OpenAI model

    # Append the embedding to the list
    chat_history_embeddings.append(gpt4all_response_embedding if model_used == "smalldog" else openai_response_embedding)

    user_sentiment = TextBlob(query).sentiment.polarity
    bot_sentiment = TextBlob(response).sentiment.polarity

    user_keywords = TextBlob(query).noun_phrases
    bot_keywords = TextBlob(response).noun_phrases

    user_keywords = (user_keywords + [None]*3)[:3]
    bot_keywords = (bot_keywords + [None]*3)[:3]

    conversation_df.loc[len(conversation_df)] = [query, response, user_sentiment, bot_sentiment, user_keywords, bot_keywords, model_used]
    
    last_response = response

    return response

# Start the conversation loop
while True:
    # Prompt the user for a query
    query = input("Please enter your query: ")
    
    # If the user types "bye", end the conversation
    if query.lower() == "bye":
        print("Goodbye!")
        break

    # Otherwise, get and print the response
    print(get_response(query))

# Save the conversation to a JSON file
conversation_df.to_json('conversation.json', orient='records', lines=True, mode='a')

# Save the embeddings to a pickle file
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(chat_history_embeddings, f)
