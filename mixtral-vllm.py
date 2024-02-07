from vllm import SamplingParams
import pandas as pd
import numpy as np
import re
import sys

class Model:
    def __init__(self, num_gens):
        from vllm import LLM

        self.n = num_gens # Num responses

        # Load the model. Tip: MPT models may require `trust_remote_code=true`.
        self.llm = LLM("mistralai/Mixtral-8x7B-Instruct-v0.1", download_dir="./", tensor_parallel_size=4)
        self.scoring_template =  """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{query}

[The Start of Assistant's Answer]
{model_resp}
[The End of Assistant's Answer]
"""

    def generate(self, user_questions, inputs):

        prompts = []
        for i in range(len(user_questions)):
            user_question = user_questions[i]
            user_question = "<s> [INST] " + user_question + " [/INST]"
            prompts.append(user_question)

        sampling_params = SamplingParams(
            n = self.n,
            max_tokens=800,
            temperature=0.5
        )
        result = self.llm.generate(prompts, sampling_params)
      
        return prompts, result

    def scoring(self, user_questions, responses):
        scoring_prompts = []
        
        for i in range(len(user_questions)):
            for j in range(self.n):
                scoring_prompts.append(self.scoring_template.format(query=user_questions[i], model_resp=responses[i][j]))

        sampling_params = SamplingParams(
            max_tokens=800,
            temperature=0.0,
        )

        result = self.llm.generate(scoring_prompts, sampling_params)

        return result

def extract_score(generation_string):
    # Use a regular expression to find the score value
    match = re.search(r'Rating: \[\[(\d+(?:\.\d+)?(?:/\d+)?)\]\]', generation_string)
    
    # If a match is found, return the score value as a string
    if match:
        return match.group(1)
    else:
        return None 

n_generations = 1
model = Model(n_generations)

json_file_path = sys.argv[1]

# Read the JSON file into a DataFrame
df = pd.read_json(json_file_path, lines=True)
# Set the number of generations (n) you want

# Display the DataFrame
questions = list(df["instruction"])
inputs = list(df['input'])

# Duplicate each row in the DataFrame n times
df = pd.DataFrame(np.repeat(df.values, n_generations, axis=0), columns=df.columns)

prompts, generations = model.generate(questions, inputs)
generations = [[x.text for x in result.outputs] for result in generations]

print("Starting Prompt scoring")
scoring_generations = model.scoring(prompts, generations)
scoring_generations_texts = [[x.text for x in result.outputs] for result in scoring_generations]

df['generations'] = sum(generations, [])
df['scoring_generations'] = sum(scoring_generations_texts, [])

df['score'] = df['scoring_generations'].apply(extract_score)

# Display the resulting DataFrame
print(df)

df.to_csv(sys.argv[1]+'processed.csv', index=False, quotechar='"')
df.to_json(sys.argv[1]+'processed.json', orient='records', lines=True)
df.to_parquet(sys.argv[1]+'processed.parquet', index=False)