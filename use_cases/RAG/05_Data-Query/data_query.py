import os
import json
import argparse
from dotenv import load_dotenv
import boto3
import botocore
from botocore.config import Config
import time
from pinecone import ServerlessSpec
from pinecone import Pinecone
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
\
load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")
DATA_DIR = os.path.join(os.path.dirname(__file__), "./jsonl")
AWS_TITAN_ENABLED = os.getenv("AWS_TITAN_ENABLED").lower() == 'true'
GCP_GEMINI_ENABLED = os.getenv("GCP_GEMINI_ENABLED").lower() == 'true'
GEMINI_PROJECT=os.getenv("GEMINI_PROJECT", "")
GEMINI_LOCATION=os.getenv("GEMINI_LOCATION", "us-central1")
GEMINI_MODEL=os.getenv("GEMINI_MODEL", "textembedding-gecko@001")
GEMINI_TEXT_GEN_MODEL=os.getenv("GEMINI_TEXT_GEN_MODEL", "gemini-1.0-pro")

def create_pinecone_connection():
    pc = Pinecone(api_key=API_KEY)
    return pc

def create_bedrock_connection():
    config = Config(connect_timeout=5, read_timeout=60, retries={"total_max_attempts": 20, "mode": "adaptive"})
    region = 'us-east-1'
    bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=region,
                endpoint_url=f'https://bedrock-runtime.{region}.amazonaws.com',
                                    config=config)
    return bedrock

def create_gemini_connection():
    vertexai.init(project=GEMINI_PROJECT, location=GEMINI_LOCATION)
    return vertexai

def titan_text_embeddings(docs: str, bedrock) -> list[float]:
    body = json.dumps({
        "inputText": docs,
    })
    
    model_id = 'amazon.titan-embed-text-v1'
    accept = 'application/json' 
    content_type = 'application/json'
    
    # Invoke model 
    response = bedrock.invoke_model(
        body=body, 
        modelId=model_id, 
        accept=accept, 
        contentType=content_type
    )
    
    # Parse response
    response_body = json.loads(response['body'].read())
    embedding = response_body.get('embedding')
    return embedding

def gemini_text_embeddings(docs: str, gemini) -> list[float]:
    model = TextEmbeddingModel.from_pretrained(GEMINI_MODEL)
    embeddings = model.get_embeddings([docs])
    return embeddings[0].values

def gemini_text_generation(prompt: str, gemini) -> str:
    model = GenerativeModel(GEMINI_TEXT_GEN_MODEL)
    output = []
    try:
        body = json.dumps(model_args(prompt))
        responses = model.generate_content(body, stream=True)
        
        for response in responses:
            print(response.text, end="")
                
    except Exception as error:
        print(f"Error during Gemini text generation: {error}")
        
    return ''.join(output)

def model_args(query):
    query_model_args = {"prompt": query, "max_tokens_to_sample": 1000, "stop_sequences": [], "temperature": 0.0, "top_p": 0.9 }
    return query_model_args

def invoke_bedrock(query, bedrock):
    output = []
    try:
        body = json.dumps(model_args(query))
        modelId = 'anthropic.claude-v2:1'
        contentType = "application/json"
        accept = "*/*"
        response = bedrock.invoke_model_with_response_stream(body=body, modelId=modelId, accept=accept, contentType=contentType)
        stream = response.get('body')
        
        i = 1
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    text = chunk_obj['completion']
                    output.append(text)
                    print(text,end='')
                    i+=1
                
    except botocore.exceptions.ClientError as error:
        
        if error.response['Error']['Code'] == 'AccessDeniedException':
            print(f"\x1b[41m{error.response['Error']['Message']}\
                    \nTo troubeshoot this issue please refer to the following resources.\
                    \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                    \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
            
        else:
            raise error
        
def construct_context(contexts: list[str]) -> str:
    chosen_sections = []
    chosen_sections_len = 0
    max_section_len = 1000
    separator = "\n"

    for text in contexts:
        text = text.strip()
        # Add contexts until we run out of space.
        chosen_sections_len += len(text) + 2
        if chosen_sections_len > max_section_len:
            break
        chosen_sections.append(text)
    concatenated_doc = separator.join(chosen_sections)
    return concatenated_doc

def create_prompt(query, context_str):
    prompt = f"""Human: Answer the following QUESTION based on the CONTEXT
    given. If you do not know the answer and the CONTEXT doesn't
    contain the answer truthfully say "I don't know".

    QUESTION:
    {query}

    CONTEXT:
    {context_str}

    Assistant:
    """

    return prompt

        

def embed(query, bedrock, gemini):
    print("Query: " + query)
    if AWS_TITAN_ENABLED:
        query_embedding = titan_text_embeddings(query, bedrock)
    elif GCP_GEMINI_ENABLED:
        query_embedding = gemini_text_embeddings(query, gemini)
    print("Vector embedding generated: " + str(query_embedding))
    return query_embedding

def search(query, bedrock, gemini, pc):
    index = pc.Index(PINECONE_INDEX_NAME)
    if AWS_TITAN_ENABLED:
        query_embedding = titan_text_embeddings(query, bedrock)
    elif GCP_GEMINI_ENABLED:
        query_embedding = gemini_text_embeddings(query, gemini)
    res = index.query(vector=query_embedding, top_k=10, namespace=PINECONE_NAMESPACE,include_metadata=True)
    print("Semantic Search results: " + str(res))
    return res

def prompt(query, bedrock, gemini, pc):
    index = pc.Index(PINECONE_INDEX_NAME)
    if AWS_TITAN_ENABLED:
        query_embedding = titan_text_embeddings(query, bedrock)
    elif GCP_GEMINI_ENABLED:
        query_embedding = gemini_text_embeddings(query, gemini)
    search_res = index.query(vector=query_embedding, top_k=10, namespace=PINECONE_NAMESPACE,include_metadata=True)
    contexts = [match.metadata["text"] for match in search_res.matches]
    context_str = construct_context(contexts=contexts)
    llm_prompt = create_prompt(query, context_str)
    print("Prompt generated: " + str(llm_prompt))
    return llm_prompt

def invoke(query, bedrock, gemini, pc):
    index = pc.Index(PINECONE_INDEX_NAME)
    if AWS_TITAN_ENABLED:
        query_embedding = titan_text_embeddings(query, bedrock)
    elif GCP_GEMINI_ENABLED:
        query_embedding = gemini_text_embeddings(query, gemini)
    search_res = index.query(vector=query_embedding, top_k=10, namespace=PINECONE_NAMESPACE,include_metadata=True)
    contexts = [match.metadata["text"] for match in search_res.matches]
    context_str = construct_context(contexts=contexts)
    llm_prompt = create_prompt(query, context_str)
    if AWS_TITAN_ENABLED:
        response = invoke_bedrock(llm_prompt, bedrock)
    elif GCP_GEMINI_ENABLED:
        response = gemini_text_generation(llm_prompt, gemini)
    return response

def main():
    parser = argparse.ArgumentParser(description="CLI for querying pinecone index data")
    parser.add_argument("action", choices=["embed", "search", "prompt", "invoke"], help="Action to perform: 'embed' generate the vector embeddings, 'search' to perform semantic search, 'prompt' to generate the prompt for LLM, 'invoke' to implement RAG")
    parser.add_argument("query", help="Query to be used for embedding, search, or RAG")
    args = parser.parse_args()
    query = args.query

    bedrock = create_bedrock_connection()
    gemini = create_gemini_connection()
    pc = create_pinecone_connection()

    if args.action == "embed":
        embed(query, bedrock, gemini)
    elif args.action == "search":
        search(query, bedrock, gemini, pc)
    elif args.action == "prompt":
        prompt(query, bedrock, gemini, pc)
    elif args.action == "invoke":
        invoke(query, bedrock, gemini, pc)

if __name__ == "__main__":
    main()