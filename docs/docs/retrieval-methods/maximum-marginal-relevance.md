# Maximum Marginal Relevance (MMR)

<iframe width="560" height="315" src="https://www.youtube.com/embed/eaZu5_dLKNk?si=k4kwnvw7HoPaY1iM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Overview

_MMR is a search type which specificies your Retrieval Method_

> The Maximal Marginal Relevance (MMR) criterion strives to reduce redundancy while maintaining query relevance in re-ranking retrieved documents - [_Source_](https://www.cs.cmu.edu/\~jgc/publication/The\_Use\_MMR\_Diversity\_Based\_LTMIR\_1998.pdf)

When it comes to retrieving documents, the majority of methods will do a similarity metric like cosine similarity, euclidean distance, or dot product. All of these will return documents that are most similar to your query/question.

However, what if you want similar documents which are also diverse from each other? That is where Maximum Marginal Relevance (MMR) steps in.

The goal is to take into account how similar retrieved documents _are to each other_ when determining which to return. In theory, you should have a well rounded, diverse set of documents.

### Why is this helpful?

1. **Complex Queries with Multiple Aspects:** When a query has several components or aspects, MMR can help retrieve a set of documents that cover all different aspects of the query, rather than just focusing on one.
2. **Avoiding Redundant Information:** In cases where the top documents returned by a simple similarity search are very similar to each other, MMR helps avoid redundant information.
3. **Content Summarization:** When creating summaries from a large set of documents, MMR can help identify key pieces of information that are both relevant and non-repetitive, which can help answer summarization questions better.
4. **Query Disambiguation:** When a query term is ambiguous or has multiple meanings, MMR can retrieve documents that represent the different senses or contexts of the term.

## MMR Code

[_See notebook_](https://github.com/gkamradt/langchain-tutorials/blob/main/data\_generation/Retrieval\_With\_MMR.ipynb)

First let's load up our keys

```python
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKey')
```

Then our LangChain imports

```python
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
```

Then let's go get our data and chunk it up

```python
# Loading a single website
loader = WebBaseLoader("http://www.paulgraham.com/wealth.html")
docs = loader.load()

# Split your website into big chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)

print (f"Your {len(docs)} documents have been split into {len(chunks)} chunks")

>> Your 1 documents have been split into 28 chunks
```

Then we'll get our embeddings and vectorstore engine going

```python
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding)
```

Then I'll make two retrievers to compare the outputs with each other:

* **Vanilla** - Regular Top K Similarity Search
* **MMR** - Do a MMR search

```python
retriever_vanilla = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 8})

retriever_mmr = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8})
```

Let's go get the docs that come from the vanilla retriever

```python
vanilla_relevant_docs = retriever_vanilla.get_relevant_documents("What is the best way to make and keep wealth?")
```

and the docs that come from the MMR retriever

```python
mmr_relevant_docs = retriever_mmr.get_relevant_documents("What is the best way to make and keep wealth?")
```

This is a long winded function to help compare the two lists together

```python
def analyze_list_overlap(list1, list2, content_attr='page_content'):
    """
    Analyze the overlap and uniqueness between two lists of objects using a specified content attribute.

    Parameters:
    list1 (list): The first list of objects to compare.
    list2 (list): The second list of objects to compare.
    content_attr (str): The attribute name of the content to use for comparison.

    Returns:
    dict: A dictionary with counts of overlapping, unique to list1, unique to list2 items,
          and total counts for each list.
    """
    # Extract unique content attributes from the lists
    set1_contents = {getattr(doc, content_attr) for doc in list1}
    set2_contents = {getattr(doc, content_attr) for doc in list2}

    # Find the number of overlapping content attributes
    overlap_contents = set1_contents & set2_contents
    overlap_count = len(overlap_contents)

    # Find the unique content attributes in each list
    unique_to_list1_contents = set1_contents - set2_contents
    unique_to_list2_contents = set2_contents - set1_contents
    unique_to_list1_count = len(unique_to_list1_contents)
    unique_to_list2_count = len(unique_to_list2_contents)

    # Use the unique content attributes to retrieve the unique objects
    unique_to_list1 = [doc for doc in list1 if getattr(doc, content_attr) in unique_to_list1_contents]
    unique_to_list2 = [doc for doc in list2 if getattr(doc, content_attr) in unique_to_list2_contents]

    # Count the total number of items in each list
    total_list1 = len(list1)
    total_list2 = len(list2)

    # Return the results in a dictionary
    return {
        'total_list1': total_list1,
        'total_list2': total_list2,
        'overlap_count': overlap_count,
        'unique_to_list1_count': unique_to_list1_count,
        'unique_to_list2_count': unique_to_list2_count,
    }
```

Then let's actually compare the lists and see what we have

```python
analyze_list_overlap(vanilla_relevant_docs, mmr_relevant_docs)
```

Cool! So the two methods returned 8 docs in total (because we asked for 8), of which, 6 of them were the same, and 2 of them were different than each other

```python
{'total_list1': 8,
 'total_list2': 8,
 'overlap_count': 6,
 'unique_to_list1_count': 2,
 'unique_to_list2_count': 2}
```

If you were to inspect the 2 MMR docs which are different you would expect they would be more diverse than the ones returned by the vanilla retriever

## References

* [Allowed Search Types (LangChain)](https://github.com/langchain-ai/langchain/blob/60d025b83be4d4f884c67819904383ccd89cff87/libs/langchain/langchain/schema/vectorstore.py#L624)
* [Code example](https://github.com/gkamradt/langchain-tutorials/blob/main/data\_generation/Retrieval\_With\_MMR.ipynb)