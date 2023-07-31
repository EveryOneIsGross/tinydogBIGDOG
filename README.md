# tinydogBIGDOG

tinydogBIGDOG is a conversational chatbot that uses both local and cloud-based language models to provide a seamless and enriched conversational experience. The chatbot is designed to first use a local language model (tinydog, powered by GPT-4All) and then, if necessary, escalate to a more powerful cloud-based model (BIGDOG, powered by OpenAI's GPT-3.5-turbo). Using essentially a student-teacher framework but at the chat end of the nn. 

## Intent

The intent of tinydogBIGDOG is to provide a high-quality conversational experience while optimizing for efficiency and cost-effectiveness. By using a local model first, we can quickly generate responses for most queries. However, for more complex or nuanced queries, we escalate to the cloud-based model to ensure we provide the best possible response. This approach allows us to balance performance and cost, providing a high-quality service while minimizing the use of more expensive cloud-based resources.

![tinydogbigdog](https://github.com/EveryOneIsGross/tinydogBIGDOG/assets/23621140/8e63570a-8dae-4754-9871-790907872c1b)

---

## Features

**Semantic Search:** tinydogBIGDOG uses semantic search to enrich the conversation context, improving the quality of the generated responses.

**Cosine Similarity:** We use cosine similarity to measure the relevance of the generated response to the user's query. If the response from the local model is not similar enough to the query, we escalate to the cloud-based model.

**Sentiment Analysis:** tinydogBIGDOG performs sentiment analysis on both the user's query and the chatbot's response, providing additional insights into the conversation.

**Keyword Extraction:** The chatbot extracts keywords from both the user's query and its own response, which can be used for further analysis or to guide the conversation.


## Consistent and Persistent Chat Agent

Despite using two different models, tinydogBIGDOG is designed to maintain a consistent conversational experience. The transition between the local and cloud-based models is seamless and invisible to the user, maintaining the illusion of a single, consistent chat agent throughout the conversation. All conversations are stored in a JSON file for future reference and analysis. Additionally, the embeddings used for semantic search are stored in a pickle file.

![KmkAHcbm](https://github.com/EveryOneIsGross/tinydogBIGDOG/assets/23621140/965be07d-cfb7-4756-8f1d-f12d97f6e2c0)

## Usage

To interact with tinydogBIGDOG, simply input your query when prompted. The chatbot will generate a response using the local model, and if necessary, escalate to the cloud-based model. The conversation continues until you type "bye".
