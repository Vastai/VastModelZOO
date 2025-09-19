# Requires vllm>=0.8.5
import torch
from vllm import LLM
import argparse

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def main(model_name: str) -> None:
    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given a web search query, retrieve relevant passages that answer the query'

    queries = [
        get_detailed_instruct(task, 'What is the capital of China?'),
        get_detailed_instruct(task, 'Explain gravity')
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    input_texts = queries + documents

    model = LLM(model=model_name, task="embed")

    outputs = model.embed(input_texts)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    scores = (embeddings[:2] @ embeddings[2:].T)
    print(scores.tolist())
    # [[0.7620252966880798, 0.14078938961029053], [0.1358368694782257, 0.6013815999031067]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen3-Embedding model")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="/logs/Qwen3-Embedding-0.6B",
        help="Path or name of the model to use"
    )
    args = parser.parse_args()
    
    main(args.model_name)
