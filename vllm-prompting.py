import argparse
import re
import pandas as pd
import numpy as np
import types
from vllm import LLM, SamplingParams

class Model:
    """
    A class representing a language model.

    Attributes:
        num_gens (int): Number of generations.
        llm (LLM): Instance of the language model.
    """

    def __init__(self, num_gens, tensor_parallel_size=4):
        """
        Initialize the Model instance.

        Args:
            num_gens (int): Number of generations.
            tensor_parallel_size (int): Size of the tensor parallelism group.
        """
        self.num_gens = num_gens
        self.llm = LLM("mistralai/Mixtral-8x7B-Instruct-v0.1", download_dir="./", tensor_parallel_size=tensor_parallel_size)

    def generate(self, user_questions, template=None, temperature=0.5):
        """
        Generate responses for user questions and optionally score them.

        Args:
            user_questions (list): List of user questions.
            template (str, optional): Template for prompts. Defaults to None.
            temperature (int, optional): Temperature for generations. Defaults to 0.5.

        Returns:
            list: List of generated responses or scoring prompts.
        """

        prompts = []
        for i in range(len(user_questions)):
            user_question = user_questions[i]
            user_question = "<s> [INST] " + user_question + " [/INST]"
            prompts.append(user_question)

        if template:
            for question in user_questions:
                prompt = template.format(query=question)
                prompts.append(prompt)
        else:
            prompts = user_questions

        sampling_params = SamplingParams(
            n = self.n,
            max_tokens=800,
            temperature=temperature,
        )
        result = self.llm.generate(prompts, sampling_params)
      
        return prompts, result

def extract_score(self, generation_string):
    """
    Extract the score from a generation string.

    Args:
        generation_string (str): Generated string.

    Returns:
        str: Extracted score.
    """
    match = re.search(r'Rating: \[\[(\d+(?:\.\d+)?(?:/\d+)?)\]\]', generation_string)
    if match:
        return match.group(1)
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description='Process JSON file with questions and inputs.')
    parser.add_argument('file_path', type=str, help='Path to the JSON file')
    parser.add_argument('n_gens', type=int, help='Number of generations', default=1)
    args = parser.parse_args()

    model = Model(args.n_gens)
    model.extract_score = types.MethodType(extract_score, model)

    # Read the JSON file into a DataFrame
    df = pd.read_json(args.file_path, lines=True)

    questions = list(df["instruction"])

    # Duplicate each row in the DataFrame n times
    df = pd.DataFrame(np.repeat(df.values, args.n_gens, axis=0), columns=df.columns)

    print("Starting generation and scoring")

    # Generate responses and scoring prompts
    model_generations = model.generate_and_score(questions)

    if isinstance(model_generations[0], list):
        # This is generated responses
        generations = model_generations
        scoring_prompts = model.generate_and_score(questions, model.scoring_template)
    else:
        # This is scoring prompts
        scoring_prompts = model_generations
        generations = model.generate_and_score(questions)

    print("Finished generation and scoring")

    # Extract scores from scoring prompts
    scores = [model.extract_score(prompt) for prompt in scoring_prompts]

    # Update DataFrame with generated responses and scores
    # df['generations'] = sum(generations, [])
    # df['scoring_generations'] = scoring_prompts
    # df['score'] = scores

    # print(df)

    # # Save processed data
    # df.to_csv(args.file_path+'processed.csv', index=False, quotechar='"')
    # df.to_json(args.file_path+'processed.json', orient='records', lines=True)
    # df.to_parquet(args.file_path+'processed.parquet', index=False)

if __name__ == "__main__":
    main()