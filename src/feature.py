import openai
import string
from typing import List
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI


def getSummaryAndTopics(doc, topic_desc: List[dict], role: str = "Singapore military analyst", prompt: str = ""):
    llm = OpenAI() # type: ignore
    doc_list = [Document(page_content=doc)]
    descript_str = ', '.join([f"{item['topic']} = {item['description']}" for item in topic_desc])
    if prompt == "":
        prompt = f"""You are a {role}. Summarised the news in 200 words, and news area ({descript_str}). In this format, Summary: <summary>, News Topic: <news topic>
        """
    chain = load_qa_chain(llm, chain_type="stuff")
    output = chain.run(input_documents= doc_list, question=prompt) # type: ignore
    summary = output.split("Summary:")[1].split("News Topic:")[0].strip()
    #remove all punctuations and uppercase news area
    topic = output.split("News Topic:")[1].strip().translate(str.maketrans('', '', string.punctuation)).upper()
    return {'summary': summary, 'topic': topic}


def getTopics(title_context, prompt_parameter : dict = {} , prompt: str = ""):
    #raise error if either prompt or prompt_parameter is empty
    if not prompt and not prompt_parameter:
        raise ValueError("Either prompt or prompt_parameter must be provided")
    llm = OpenAI() # type: ignore
    doc = [Document(page_content=title_context)]
    if not prompt:      
        role = prompt_parameter['role']
        top_n = prompt_parameter['top_n']
        topic_length = prompt_parameter['topic_length']
        prompt = f"""I am a {role}. List down {top_n} most impactful/concerning and distinctive topics (max {topic_length} words) from the news. In this format, Topics: [<topic 1> : <topic 1 description>, ..., <topic {top_n}> : <topic {top_n} description>]
        """
    chain = load_qa_chain(llm, chain_type="stuff")
    output = chain.run(input_documents= doc, question=prompt) # type: ignore
    topics = output.split("Topics:")[1].strip().replace("[","").replace("]","").replace("'","").split(", ")
    topic_desc = [{'topic': topic.split(": ")[0], 'description': topic.split(": ")[1]} for topic in topics]

    return topic_desc