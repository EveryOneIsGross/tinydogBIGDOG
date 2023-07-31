import numpy as np
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from gpt4all import GPT4All, Embed4All
import openai
import os
import dotenv
import pandas as pd
import pickle
import logging

logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

dotenv.load_dotenv()

class GPT4AllChatbot:
    def __init__(self):
        self.model = GPT4All(
            model_name=('c:\\AI_MODELS\\llama2_7b_chat_uncensored.ggmlv3.q4_0.bin')
        )

        self.embedder = Embed4All()

    def generate_response(self, query, last_response=None):
        prompt = query
        if last_response:
            # Use the last response to provide context
            prompt += ' ' + last_response
        # Reinforce the user's query
        prompt += ' ' + query
        logging.debug(f'Prompt: {prompt}')
        with self.model.chat_session():
            return self.model.generate(
                prompt=prompt,
                #top_k=1,
                #n_predict=800,
                temp=0.6
            )
        
    def embed(self, text):
        return self.embedder.embed(text)



class OpenAIChatbot:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def generate_response(self, query, last_response=None):
        messages = [
            {"role": "system", "content": "You are cloud hosted query chatbot. Do not write lists, answer as directly as possible."},
            {"role": "user", "content": query}
        ]
        if last_response:
            # Use the last response to provide context
            messages.append({"role": "assistant", "content": last_response})
        # Reinforce the user's query
        messages.append({"role": "user", "content": query})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=200,
            messages=messages
        )
        return response['choices'][0]['message']['content']


class SentenceTransformerSummarizer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def summarize(self, text):
        # Split the text into sentences
        sentences = text.split('.')

        # If the text consists of a single sentence or is empty, return the text as the summary
        if len(sentences) <= 1:
            return text

        # Get embeddings for each sentence
        embeddings = self.model.encode(sentences, convert_to_tensor=True)

        # Check if embeddings are not empty and have the correct shape
        if embeddings is None or embeddings.shape[0] != len(sentences):
            return text

        # Use the mean pooling method to get a single vector for the entire text
        mean_embedding = util.pytorch_cos_sim(embeddings, embeddings).mean(dim=0)

        # Check if mean_embedding is not None and has the correct shape
        if mean_embedding is None or mean_embedding.shape[0] != embeddings.shape[1]:
            return text

        # Find the sentence that is most similar to the mean embedding
        most_similar_sentence = util.pytorch_cos_sim(mean_embedding, embeddings).argmax()

        # Check if most_similar_sentence is not None and is a valid index
        if most_similar_sentence is None or most_similar_sentence.item() >= len(sentences):
            return text
        logging.debug(f'Sentence Transformer similarity: {util.pytorch_cos_sim(mean_embedding, embeddings)[0][most_similar_sentence]}')

        # Return the most similar sentence as the summary
        return sentences[most_similar_sentence.item()]

    


# Initialize the chatbots
gpt4all_chatbot = GPT4AllChatbot()
openai_chatbot = OpenAIChatbot()
boneSUMMARY = SentenceTransformerSummarizer()

# Set the threshold for the semantic search, high values will recall more conversations, low values will recall fewer conversations
semantic_search_threshold = 1.0

# Set the threshold for the cosine similarity score between the query and the chatbot's response
response_similarity_threshold = 0.2 # high values will choose gpt4all_chatbot, low values will choose openai_chatbot

# Initialize the chat history and embeddings
chat_history = []

# Check if the JSON file exists
if os.path.exists('conversation.json'):
    # If it exists, load the existing conversations
    conversation_df = pd.read_json('conversation.json', orient='records', lines=True)
else:
    # If it doesn't exist, create a new DataFrame
    conversation_df = pd.DataFrame(columns=['User_Input', 'Bot_Response', 'User_Sentiment', 'Bot_Sentiment', 'User_Keywords', 'Bot_Keywords', 'Model'])

# Check if the embeddings pickle file exists
if os.path.exists('embeddings.pkl'):
    # If it exists, load the existing embeddings
    with open('embeddings.pkl', 'rb') as f:
        chat_history_embeddings = pickle.load(f)
else:
    # If it doesn't exist, create a new list
    chat_history_embeddings = []
    
def recall_memory(query_keywords):
    # Initialize a counter for recalled conversations
    recalled_conversations = 0
    recalled_memory = ""

    # Iterate over the conversation DataFrame
    for index, row in conversation_df.iterrows():
        # If the query keywords overlap with the user keywords in the row
        if set(query_keywords).intersection(set(row['User_Keywords'])):
            # Summarize the past conversation
            summary = boneSUMMARY.summarize(f"{row['User_Input']} {row['Bot_Response']}")
            logging.debug(f'Summary: {summary}')

            # Append the summary to the recalled memory
            recalled_memory += summary + "\n"

            # Increment the counter
            recalled_conversations += 1

            # If we've recalled two conversations, stop recalling
            if recalled_conversations >= 2:
                break 

    return recalled_memory


def semantic_search(query_embedding, history_embeddings):
    # Compute cosine similarity between the query embedding and each history embedding
    similarities = cosine_similarity(query_embedding, history_embeddings)
    # Get the index of the most similar history embedding
    most_similar_index = np.argmax(similarities)
    logging.debug(f'Semantic search similarity: {similarities[0][most_similar_index]}')
    return chat_history[most_similar_index]



last_response = ""

def get_response(query):
    global last_response
    logging.debug(f'User query: {query}')
    # Extract keywords from the query
    query_keywords = TextBlob(query).noun_phrases

    # Recall past conversations based on the query keywords
    recalled_memory = recall_memory(query_keywords)

    # Incorporate the recalled memory into the query
    query += ' ' + recalled_memory

    query_embedding = np.array(gpt4all_chatbot.embed(query)).reshape(1, -1)

    if chat_history:
        similar_response = semantic_search(query_embedding, np.array(chat_history_embeddings))
        if similar_response >= semantic_search_threshold:
            # Summarize the similar response
            similar_response = boneSUMMARY.summarize(similar_response)
            query += ' ' + similar_response

    gpt4all_response = gpt4all_chatbot.generate_response(query, last_response)
    logging.debug(f'GPT4All response: {gpt4all_response}')
    gpt4all_response_embedding = np.array(gpt4all_chatbot.embed(gpt4all_response)).reshape(1, -1)

    similarity_score_gpt4all = cosine_similarity(query_embedding, gpt4all_response_embedding)

    # Initialize model_used
    model_used = None

    if similarity_score_gpt4all >= response_similarity_threshold:
        response = gpt4all_response
        model_used = "smalldog"  # GPT-4All model
    else:
        openai_response = openai_chatbot.generate_response(query, last_response)
        logging.debug(f'OpenAI response: {openai_response}')
        openai_response_embedding = np.array(gpt4all_chatbot.embed(openai_response)).reshape(1, -1)
        response = openai_response
        model_used = "BIGDOG"  # OpenAI model

    # Append the embedding to the list
    chat_history_embeddings.append(gpt4all_response_embedding if model_used == "smalldog" else openai_response_embedding)

    # Extract keywords from the query and response after the response has been generated
    user_keywords = TextBlob(query).noun_phrases
    bot_keywords = TextBlob(response).noun_phrases

    user_sentiment = TextBlob(query).sentiment.polarity
    bot_sentiment = TextBlob(response).sentiment.polarity

    user_keywords = (user_keywords + [None]*3)[:3]
    bot_keywords = (bot_keywords + [None]*3)[:3]

    logging.debug(f'User keywords: {user_keywords}')
    logging.debug(f'Bot keywords: {bot_keywords}')

    conversation_df.loc[len(conversation_df)] = [query, response, user_sentiment, bot_sentiment, user_keywords, bot_keywords, model_used]
    
    last_response = response

    return response


# Initialize the last response
last_response = None

# Initialize the chat history embeddings
chat_history_embeddings = []

# Initialize the conversation DataFrame
conversation_df = pd.DataFrame(columns=['User_Input', 'Bot_Response', 'User_Sentiment', 'Bot_Sentiment', 'User_Keywords', 'Bot_Keywords', 'Model_Used'])

# Start the conversation loop
while True:
    # Get the user's input
    query = input("You: ")

    # If the user wants to quit, break the loop
    if query.lower() in ['quit', 'exit', 'stop', 'bye', 'goodbye', 'end']:
        break

    # Get the bot's response
    response = get_response(query)

    # Print the bot's response
    print(f"Bot: {response}")


# Save the conversation to a JSON file
conversation_df.to_json('conversation.json', orient='records', lines=True, mode='a')

# Save the embeddings to a pickle file
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(chat_history_embeddings, f)
