# Top-K Similarity Search

<!-- {% embed url="https://youtu.be/8YtfmOqj28c" %} -->

## Overview

_Top-K Similarity search type which specifies how you should retrieve documents from your knowledge base. This is type of a_ [_retrieval method_](./)_._

Top-K Similarity search is the process of pulling "K" number of documents from your knowledge base. Its aim is to get you the most _similar_ documents to a query according to your similarity metric.

There are two concepts which are important

* **Similarity Metric** - This is the equation or metric you'll use to compare the embedding of your query to the embeddings of your corpus of documents. The most common similarity metrics are [cosine](https://www.youtube.com/watch?v=e9U0QAFbfLI), dotproduct, and euclidean. The smaller the metric, the more similar two embeddings are expected to be.
* **Number of documents to retrieve** - This specifies the number of documents to retrieve. It may seem like more is better, but remember, it's important to keep the signal to noise ratio of your retrieval process high. The more documents you retrieve the more likely it is you'll overwhelm your model.

You'll often see us refer to this method as "vanilla retrieval." This is a delicious example of the basic retrieval method which powers many chatbots and question/answer tools. Further, it's commonly referred to as the "Hello world" example of doing retrieval.

<!-- <figure><img src="../.gitbook/assets/TopKSimilaritySearch.gif" alt=""><figcaption></figcaption></figure> -->

### Why is this helpful?

1. **Simple** - This method has very few moving parts while taking advantage of powerful language models
2. **Foundational** - This method is the start point for many advanced methods of retrieval

## Top-K Similarity Code

[_See notebook_](https://github.com/gkamradt/langchain-tutorials/blob/main/data\_generation/Ask%20A%20Book%20Questions.ipynb)

<!-- {% code overflow="wrap" %} -->
```python
# PDF Loaders. If unstructured gives you a hard time, try PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()
```
<!-- {% endcode %} -->

#### Load your data

Next let's load up some data. I've put a few 'loaders' on there which will load data from different locations. Feel free to use the one that suits you. The default one queries one of Paul Graham's essays for a simple example. This process will only stage the loader, not actually load it.

<!-- {% code overflow="wrap" %} -->
```python
loader = TextLoader(file_path="../data/PaulGrahamEssays/vb.txt")

## Other options for loaders 
# loader = PyPDFLoader("../data/field-guide-to-data-science.pdf")
# loader = UnstructuredPDFLoader("../data/field-guide-to-data-science.pdf")
# loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")
```
<!-- {% endcode %} -->

Then let's go ahead and actually load the data.

```python
data = loader.load()
```

Then let's actually check out what's been loaded

<!-- {% code overflow="wrap" %} -->
```python
# Note: If you're using PyPDFLoader then it will split by page for you already
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your sample document')
print (f'Here is a sample: {data[0].page_content[:200]}')

> You have 1 document(s) in your data
There are 9155 characters in your sample document
Here is a sample: January 2016Life is short, as everyone knows. When I was a kid I used to wonder
about this. Is life actually short, or are we really complaining
about its finiteness?  Would we be just as likely to fe
```
<!-- {% endcode %} -->

#### Chunk your data up into smaller documents

While we could pass the entire essay to a model w/ long context, we want to be picky about which information we share with our model. The better signal to noise ratio we have the more likely we are to get the right answer.

The first thing we'll do is chunk up our document into smaller pieces. The goal will be to take only a few of those smaller pieces and pass them to the LLM.

<!-- {% code overflow="wrap" %} -->
```python
# We'll split our data into chunks around 500 characters each with a 50 character overlap. These are relatively small.

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(data)
```
<!-- {% endcode %} -->

```python
# Let's see how many small chunks we have
print (f'Now you have {len(texts)} documents')
```

#### Create embeddings of your documents to get ready for semantic search

Next up we need to prepare for similarity searches. The way we do this is through embedding our documents (getting a vector per document).

This will help us compare documents later on.

```python
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
```

Check to see if there is an environment variable with you API keys, if not, use what you put below

```python
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
```

Then we'll get our embeddings engine going. You can use whatever embeddings engine you would like. We'll use OpenAI's ada today.

```python
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
```

I like Chroma because it's local and easy to set up without an account.

First we'll pass our texts to Chroma via `.from_documents`, this will 1) embed the documents and get a vector, then 2) add them to the vectorstore for retrieval later.

```python
# load it into Chroma
vectorstore = Chroma.from_documents(texts, embeddings)
```

Let's test it out. I want to see which documents are most closely related to a query.

```python
query = "What is great about having kids?"
docs = vectorstore.similarity_search(query)
```

Then we can check them out. In theory, the texts which are deemed most similar should hold the answer to our question. But keep in mind that our query just happens to be a question, it could be a random statement or sentence and it would still work.

<!-- {% code overflow="wrap" %} -->
```python
# Here's an example of the first document that was returned
for doc in docs:
    print (f"{doc.page_content}\n")

> jabs into your consciousness like a pin.The things that matter aren't necessarily the ones people would
call "important."  Having coffee with a friend matters.  You won't
feel later like that was a waste of time.One great thing about having small children is that they make you
spend time on things that matter: them. They grab your sleeve as
you're staring at your phone and say "will you play with me?" And

the question, and the answer is that life actually is short.Having kids showed me how to convert a continuous quantity, time,
into discrete quantities. You only get 52 weekends with your 2 year
old.  If Christmas-as-magic lasts from say ages 3 to 10, you only
get to watch your child experience it 8 times.  And while it's
impossible to say what is a lot or a little of a continuous quantity
like time, 8 is not a lot of something.  If you had a handful of 8

January 2016Life is short, as everyone knows. When I was a kid I used to wonder
about this. Is life actually short, or are we really complaining
about its finiteness?  Would we be just as likely to feel life was
short if we lived 10 times as long?Since there didn't seem any way to answer this question, I stopped
wondering about it.  Then I had kids.  That gave me a way to answer

done that we didn't.  My oldest son will be 7 soon.  And while I
miss the 3 year old version of him, I at least don't have any regrets
over what might have been.  We had the best time a daddy and a 3
year old ever had.Relentlessly prune bullshit, don't wait to do things that matter,
and savor the time you have.  That's what you do when life is short.Notes[1]
At first I didn't like it that the word that came to mind was
one that had other meanings.  But then I realized the other meanings
```
<!-- {% endcode %} -->

#### Query those docs to get your answer back

Great, those are just the docs which should hold our answer. Now we can pass those to a LangChain chain to query the LLM.

We could do this manually, but a chain is a convenient helper for us.

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
```

Here we'll initalize our LLM and our chain

```python
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
```

Then let's go ask the same question as before

```python
query = "What is great about having kids?"
docs = vectorstore.similarity_search(query)
```

<!-- {% code overflow="wrap" %} -->
```python
chain.run(input_documents=docs, question=query)

> 'One great thing about having kids is that they make you spend time on things that matter. They remind you to prioritize the important things in life, like spending quality time with them. Having kids can also bring a sense of joy and fulfillment as you watch them grow and experience new things.'
```
<!-- {% endcode %} -->

Awesome! We just went and queried an external data source!

## References

* [load\_qa\_chain on github](https://github.com/langchain-ai/langchain/blob/14799b139ad2a5cd78450d0c1be8bc8c78db38cf/libs/langchain/langchain/chains/question\_answering/\_\_init\_\_.py#L219)
* [rag\_prompt on LangChain hub](https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=50995362-9ea0-4378-ad97-b4edae2f9f22)