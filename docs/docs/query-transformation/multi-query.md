# Multi-Query 🦜🔗

<iframe width="560" height="315" src="https://www.youtube.com/embed/VYmiVkalgRc?si=zCM2jVX8dsb0u7j7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Overview

_Multi-Query is an advanced method of the Query Transformation stage of retrieval_

In traditional retrieval you generally will supply 1 question or query to your database and using a similarity measure, you'll get 4-5 documents back.

However the **multi-query** method will generate multiple queries (hence the name) and then use each one of the queries to get similar docs.

The goal is that the docs returned represent a more well rounded context for your LLM to work with.

![Retrieval Basics](img/MultiQuery.gif)

### Why is this helpful?

Generally builders will use Multi-Query for two main reasons: Enhance a suboptimal query & Expand a results set

#### **Enhance a suboptimal query**

Users don't always give the best queries, we can't blame them though - They are just trying to user your product, not construct the perfect query.

To help with this, we turn to the multi-query method to help us fill in any gaps to a users query.

#### **Expand a results set**

With multiple queries, you'll likely get more results back from your database. The aim of multi-query is to have an expanded results sets which might be able to answer questions better than docs from a single query.

These results will be deduplicated (in case the same document comes back multiple times) and then used as context in your final prompt.

## Multi-Query Code

First load up your keys

```python
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKey')
```

Then load up your documents. We'll move quickly through this part. Head over to our document loaders section if you'd like to see more.

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader

# Loading a single website
loader = WebBaseLoader("http://www.paulgraham.com/superlinear.html")

docs = loader.load()

# Split it up
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)

# Embed and store your docs/vectors
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding)
```

Now let's set up our Multi-Query Retriever

```python
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

# Set logging for the queries
import logging

# Set up logging to see your queries
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
```

Now we'll pose an original question or query to

```python
# Your original question
question = "What are the fundamental use cases of superlinear returns?"
llm = ChatOpenAI(temperature=0)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)
```

Then once we actually ask for the relevant docs, we'll see the multiple other questions that were generated from our original question

```python
unique_docs = retriever_from_llm.get_relevant_documents(query=question)

>>
Generated queries:
[
    '1. Can you provide examples of the primary applications where superlinear returns are observed?',
    '2. In what scenarios do we typically see superlinear returns being utilized?',
    '3. Could you list some fundamental use cases that demonstrate the concept of superlinear returns?'
]
```

The additional queries that were generated will now be used in addition to our original query to retrieve documents.

The documents returned will be deduplicated and passed along to the rest of your application.

Pretty cool huh?

### Editing The Prompt To Generate Additional Queries

If you wanted to edit the prompt template that is being used by the MultiQuery Retriever you can do that when you first create it. See the original prompt that is being used [here](https://github.com/langchain-ai/langchain/blob/60d025b83be4d4f884c67819904383ccd89cff87/libs/langchain/langchain/retrievers/multi\_query.py#L38).

```python
prompt_template = """You are an AI language model assistant.

Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database.

By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations  of distance-based similarity search.

Provide these alternative questions separated by newlines.

Original question: {question}"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["question"]
)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm, prompt=PROMPT
)
```

## References

* [Full notebook with code](https://github.com/gkamradt/langchain-tutorials/blob/main/data\_generation/Advanced%20Retrieval%20With%20LangChain.ipynb)
* [LangChain Documentation](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever)
  * [Prompt used by default](https://github.com/langchain-ai/langchain/blob/60d025b83be4d4f884c67819904383ccd89cff87/libs/langchain/langchain/retrievers/multi\_query.py#L38)