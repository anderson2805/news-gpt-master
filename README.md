# Newsletter Generation Tool
This tool is designed to generate a weekly newsletter containing summaries of news articles based on different topics. It is built using Python and the OpenAI GPT-3.5 language model.

## Dependencies
- openai
- string
- typing
- pandas
- tqdm
- langchain

## Model Download
At model/universal-sentence-encoder-large/5/download_use_5.bat
will download model from : https://tfhub.dev/google/universal-sentence-encoder-large/5
and unzipped the file in same directory.

## Features
### feature.py
This file contains two functions:

- getSummaryAndTopics(doc, topic_desc: List[dict], role: str = "Singapore military analyst", prompt: str = ""): This function takes in a document, a list of topic descriptions, a role, and a prompt as inputs. It uses OpenAI's GPT-3.5 to generate a summary and a topic based on the inputs provided.

- getTopics(title_context): This function takes in a document and uses OpenAI's GPT-3.5 to generate a list of topics and their descriptions based on the document.

### utils.py

This file contains a single function:
- concatContext(df_column): This function takes in a Pandas dataframe and returns a concatenated string of all the values in the dataframe.

### generate.py
To use this script, you need to have an OpenAI API key, which can be obtained from the OpenAI website. Once you have the key, set it as an environment variable named "OPENAI_API_KEY".

The script takes as input a list of dictionaries, where each dictionary contains the title and content description of a news article. The output is a list of dictionaries containing the 'topic', 'prompt', and 'response' for each section.

This file contains two functions:

- generate_section(context, role: str = "", topic: str = "", form: str = "5 non-numbered point forms", prompt: str = ""): This function takes in a context, a role, a topic, a form type, and a prompt as inputs. It uses OpenAI's GPT-3.5 to generate a section for the newsletter based on the inputs provided.

- generate_newsletter(query_dict: List[dict], topic_desc: List[dict] = topic_desc, role: str = role, form_type: dict = form_type, max_articles_section: int = 10, template : str = template) -> List[dict]: This function takes in a list of dictionaries containing the titles and contents of news articles, as well as optional inputs for the topic descriptions, role, form type, maximum number of articles per section, and template type. It uses OpenAI's GPT-3.5 to generate a newsletter based on the inputs provided.

    ### Parameters
    - query_dict: A list of dictionaries where each dictionary contains a 'title' and 'contentdescription' of a news article.
    - topic_desc: A list of dictionaries containing the topics, descriptions, and prompts for each section. Default is topic_desc variable defined in the function.
    - role: The role of the person generating the newsletter. Default is Singapore military analyst.
    - form_type: The format of the summary. Default is a bullet point format.
    - max_articles_section: The maximum number of articles to include in each section. Default is 10.
    - template: A predefined template for the newsletter. Default is predefined.

## Usage
To use this tool, simply import the necessary functions from the Python files and call them with the appropriate inputs.
1. Prepare a list of dictionaries containing the 'title' and 'contentdescription' of each news article. Save it as a JSON file.
1. Modify the values of topic_desc, role, form_type, and template variables in the newsletter_generator.py script if needed.
1. Run the script: python newsletter_generator.py.

For example, to generate a summary and a topic based on a document:
```
import json
from newsletter_generator import generate_newsletter

# Load the news articles
with open('./data/query2002_2602.json') as f:
    data = json.load(f)

# Generate the newsletter
newsletter = generate_newsletter(data)

# Print the newsletter
print(newsletter)

```