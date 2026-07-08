from langsmith import Client
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
from langsmith.evaluation import evaluate
load_dotenv()

# client = Client()

# dataset = client.create_dataset(
#     dataset_name="eval_datasets",
#     description="Dataset for evaluating QA models"
# )

# client.create_examples(
#     inputs=[
#         {"input": "What is the capital of France?"},
#         {"input": "Who wrote 'Pride and Prejudice'?"},
#         {"input": "What is the largest planet in our solar system?"},
#         {"input": "What is the boiling point of water?"},
#         {"input": "Who painted the Mona Lisa?"},
#     ],
#     outputs=[
#         {"output": "The capital of France is Paris."},
#         {"output": "Jane Austen wrote 'Pride and Prejudice'."},
#         {"output": "The largest planet in our solar system is Jupiter."},
#         {"output": "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."},
#         {"output": "The Mona Lisa was painted by Leonardo da Vinci."},
#     ],
#     dataset_id=dataset.id,
# )


# define metrics llm as a judge
def target(input:dict):
    response = judge.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": f"Please answer the following question: {input}"}
        ],
        response_format={"type": "json_object"}
    )
    return {
        "output": response.choices[0].message.content
    }


judge = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def corretness(input:dict,output:dict,reference_outputs:dict):
    prompt = f"""
    You are a judge for evaluating the correctness of answers to questions. 
    Given the following question, answer, and reference answer, determine if the answer is correct or not.
    
    Question: {input}
    Answer: {output}
    Reference Answer: {reference_outputs}

    Return ONLY valid JSON.
    {{
        "score": 0-10,
        "reasoning": "Provide reasoning for the score.",
        "pass": true/false

    }}
    
    """
    response = judge.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    result =  json.loads(response.choices[0].message.content)


    return {
        "key": "correctness",
        "value": result["score"],
        "reasoning": result["reasoning"],
        "pass": result["pass"]
    }


eval = evaluate(target=target, 
                data='eval_datasets',
                evaluators=[corretness])