import openai
import string
from typing import List
import pandas as pd
from feature import getSummaryAndTopics
from utils import concatContext
from tqdm import tqdm
tqdm.pandas()

MAX_ARTICLE_TO_ANALYSE = 40

topic_desc = [{'topic': 'LOCAL', 'description': 'news about Singapore', 'prompt': ""}, 
              {'topic': 'GLOBAL', 'description': 'news about other countries'}, 
              {'topic': 'PROCUREMENT', 'description': 'news about procurement/purchase'}, 
              {'topic': 'TECHNOLOGY', 'description': 'news about technology'}]
role = 'Singapore military analyst'
# numbered points, paragraphs, words summary, words synopsis
form_type = {'type': 'bullet points', 'parameter': None}
template = 'predefined'


def generate_section(context, role: str = "", topic: str = "", form: str = "5 non-numbered point forms", prompt: str = ""):
    if not (role or topic) and not prompt:
        raise ValueError(
            "Either 'role' and 'topic' or 'prompt' must be provided.")
    else:
        default_prompt = f"You are a {role}. Based on the context, generate a weekly news summary for management about most impactful/concerning/dispute {topic} news in {form}. At the end of generation reply <END>. In this format, {topic} News: <summaries>."
        if not prompt:
            prompt = default_prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Singapore military analyst."},
                {"role": "user", "content": f"Context: {context} \n {prompt}"}
            ]
        )
        
        return {'topic': topic, 'prompt': prompt, 'response': response['choices'][0]['message']['content'].split("<END>")[0]} # type: ignore


def generate_newsletter(query_dict: List[dict], topic_desc: List[dict] = topic_desc,
                        role: str = role, form_type: dict = form_type, max_articles_section: int = 10, template: str = template) -> List[dict]:
    """
    Generate a weekly newsletter containing summaries of news articles based on different topics.

    Parameters:
        query_dict (List[dict]): A list of dictionaries where each dictionary contains a 'title' and 'contentdescription' of a news article.
        topic_desc (List[dict], optional): A list of dictionaries containing the topics, descriptions, and prompts for each section. Default is `topic_desc` variable defined in the function.
        role (str, optional): The role of the person generating the newsletter. Default is `Singapore military analyst`.
        form_type (dict, optional): The format of the summary. Default is a bullet point format.
        max_articles_section (int, optional): The maximum number of articles to include in each section. Default is `10`.
        template (str, optional): A predefined template for the newsletter. Default is `predefined`.

    Returns:
        List[dict]: A list of dictionaries containing the 'topic', 'prompt', and 'response' for each section.

    TODO: Customise the prompt based on the template = "Topic", need to adjust getsummaryandtopics function
    """
    
    newsletter = []
    queryDf = pd.DataFrame(query_dict)[:MAX_ARTICLE_TO_ANALYSE]
    # apply getsummaryregion to each row of the queryDf title and contentdescription columns, output will be a dictionary with keys 'summary' and 'news_area, and store the result in the two columns
    if template == 'predefined':
        topic_desc_str = ', '.join(
            [f"{item['topic']} = {item['description']}" for item in topic_desc])
    elif template == 'topic':
        topic_desc_str = ', '.join([f"{item['topic']}" for item in topic_desc])
    queryDf[['summary', 'topic']] = queryDf.progress_apply(lambda x: pd.Series(
        getSummaryAndTopics(x['contentdescription'], topic_desc_str=topic_desc_str)), axis=1)

    for desc in topic_desc:
        sectionDf = queryDf[queryDf['topic'].str.contains(desc['topic'].upper())]

        form_parameter = form_type['parameter'] if form_type['parameter'] is not None else min(
            5, len(sectionDf))
        form = f"{form_parameter} {form_type.get('type', '')}"

        if(len(sectionDf) == 0):
            section_dict = {'topic': desc['topic'], 'prompt': 'NA',
                            'response': 'No news to report.'}
        else:
            # if we are requesting reference from ChatGPT response, we need to provide it in context
            context = concatContext(
                sectionDf.iloc[:max_articles_section][['summary']])
            section_dict = generate_section(
                context, role=role, topic=desc['topic'], form=form, prompt=desc.get('prompt', ''))
        newsletter.append(section_dict)
    return newsletter


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
    print(generate_newsletter(data))
