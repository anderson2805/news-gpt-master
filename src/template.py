
from feature import getTopics
from generate import generate_newsletter
from utils import concatContext
import pandas as pd

role = 'Singapore military analyst'
form_type = {'type': 'words synopsis', 'parameter': 500} #numbered points, paragraphs, words summary, words synopsis
template = 'topic'
topic_prompt = {'role': 'Singapore military analyst', 'top_n': 3, 'topic_length': 5}



if __name__ == "__main__":
    import json
    import os

    # from dotenv import load_dotenv
    # load_dotenv()
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    # or
    OPEN_API_KEY = ''
    # Set the environment variable
    os.environ['OPENAI_API_KEY'] = OPEN_API_KEY
    
    with open('./data/query1302_1902.json') as f:
        data = json.load(f)
    dataDf = pd.DataFrame(data)

    topic_desc = getTopics(title_context = concatContext(dataDf[['articletitle']]), prompt_parameter =topic_prompt)
    print(topic_desc)

    newsletter = generate_newsletter(data, topic_desc=topic_desc, role=role, form_type=form_type, template=template)
    print(newsletter)