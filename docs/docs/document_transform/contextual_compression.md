---
description: '"The Fluff Remover" - Condense retrieved information for final context'
---

# Contextual Compression

<!-- {% embed url="https://youtu.be/tsxEycc4tB0" %} -->

## Overview

_Contextual Compression is method of the to_ [_transform_](./) _your retrieved documents_

Contextual Compression refers to the process of extracting information from your retrieved docs that you think will be relevant to your final answer. It's all about increasing the signal-to-noise ratio.

Contextual compression works by iterating over your retrieved documents, then passing them through a LLM which will extract information according to context you specify in a prompt.

You're _compressing_ your final docs based on _context_ you give it, "contextual compression" get it?

The key component here is the "compressor" which will do our extraction/compressing for us.

Say you wanted to know everything about bananas, but you retrieved document talks about apples, oranges, and bananas. The compressor will pull out everything about bananas and then pass it on to your final prompt for a response.

<!-- <figure><img src="../.gitbook/assets/ContextualCompression (1).gif" alt=""><figcaption></figcaption></figure> -->

### Why is this helpful?

The problem with vanilla retrieval is once you chunk your original documents, your chunks may have multiple semantic topics held within them. If you were to pass those multiple topics to the LLM, you may confuse it.

Contextual Compression is helpful when you want to try and refine your retrieved documents further.

### What are the downsides?

You'll be doing an addition API call(s) based on the number of retrieved documents you have. This will increase costs and latency of your application.

## Contextual Compression Code

Let's start by grabbing our keys

```python
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKey')
```

Then let's import the packages we'll need

```python
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0, model='gpt-4')
```

and load up some data

```python
# Loading a single website
loader = WebBaseLoader("http://www.paulgraham.com/superlinear.html")
docs = loader.load()

# Split your website into big chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)

print (f"Your {len(docs)} documents have been split into {len(chunks)} chunks")

>> Your 1 documents have been split into 5 chunks
```

Then we'll get our embeddings and vectorstore ready

```python
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding)
```

We first need to set up our compressor, it's cool that it's a separate object because that means you can use it elsewhere outside this retriever as well.

```python
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vectordb.as_retriever(search_kwargs={"k": 2}))
```

Great, now that we have our compressor set up, let's try vanilla retrieval first to see what we have

```python
relevant_docs = compression_retriever.base_retriever.get_relevant_documents("What do people think is a flaw of capitalism?")
print (f"You have {len(relevant_docs)} relevant docs")
print (relevant_docs[0].page_content[:500])
print (f"Your first document has length: {len(relevant_docs[0].page_content)}")

>> You have 2 relevant docs
gradual improvements in technique, not the discoveries of a few
exceptionally learned people.[3]
It's not mathematically correct to describe a step function as
superlinear, but a step function starting from zero works like a
superlinear function when it describes the reward curve for effort
by a rational actor. If it starts at zero then the part before the
step is below any linearly increasing return, and the part after
the step must be above the necessary return at that point or no one
would bo
Your first document has length: 3920
```

Great, now let's see what this same document looks like, but compressed based on the context we give it.

We are going to pass a question to the compressor and with that question we will compress the doc. The cool part is this doc will be _contextually compressed_, meaning the resulting file will only have the information relevant to the question.

```python
compressor.compress_documents(documents=relevant_docs, query="What do people think is a flaw of capitalism?")

>> [Document(page_content='"It\'s unclear exactly what advocates of "equity" mean by it. They seem to disagree among themselves. But whatever they mean is probably at odds with a world in which institutions have less power to control outcomes, and a handful of outliers do much better than everyone else.It may seem like bad luck for this concept that it arose at just the moment when the world was shifting in the opposite direction, but I don\'t think this was a coincidence. I think one reason it arose now is because its adherents feel threatened by rapidly increasing variation in performance."', metadata={'language': 'No language found.', 'source': 'http://www.paulgraham.com/superlinear.html', 'title': 'Superlinear Returns'}),
 Document(page_content="Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true. But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented. We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity. In all of these, the rich get richer.", metadata={'language': 'No language found.', 'source': 'http://www.paulgraham.com/superlinear.html', 'title': 'Superlinear Returns'})]
```

Great so we had a long document, now we have a shorter document with more dense information. Great for getting rid of the fluff. Let's try it out on our essays

```python
question = "What do people think is a flaw of capitalism?"
compressed_docs = compression_retriever.get_relevant_documents(question)
```

We now have docs but they are shorter and only contain the information that is relevant to our query.

Let's put it in our prompt template again.

```python
prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
```

```python
llm.predict(text=PROMPT.format_prompt(
    context=compressed_docs,
    question=question
).text)

>> 'Some people think that superlinear returns for performance, where a handful of outliers do much better than everyone else, is a flaw of capitalism.'
```

## References

* [Full notebook with code](https://github.com/gkamradt/langchain-tutorials/blob/main/data\_generation/Advanced%20Retrieval%20With%20LangChain.ipynb)
* [LangChain Documentation](https://python.langchain.com/docs/modules/data\_connection/retrievers/multi\_vector)