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

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")
DATA_DIR = os.path.join(os.path.dirname(__file__), "./jsonl")

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
    max_section_len = 5000
    separator = "\n"

    for text in contexts:
        text = text.strip()
        # Add contexts until we run out of space.
        chosen_sections_len += len(text) + 2
        if chosen_sections_len > max_section_len:
            break
        chosen_sections.append(text)
    concatenated_doc = separator.join(chosen_sections)
    print(
        f"With maximum sequence length {max_section_len}, selected top {len(chosen_sections)} document sections: \n{concatenated_doc}"
    )
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
    print("Query Prompt:", str(prompt))
    return prompt

        

def embed(query, bedrock):
    print("Query: " + query)
    query_embedding = titan_text_embeddings(query, bedrock)
    print("Vector embedding generated: " + query_embedding)
    return query_embedding

def search(query, bedrock, pc):
    index = pc.Index(PINECONE_INDEX_NAME)
    query_embedding = embed(query, bedrock)
    res = index.query(vector=query_embedding, top_k=10, namespace=PINECONE_NAMESPACE,include_metadata=True)
    print("Semantic Search results: " + res)
    return res

def invoke(query, bedrock, pc):
    search_res = search(query, bedrock, pc)
    contexts = [match.metadata["text"] for match in search_res.matches]
    context_str = construct_context(contexts=contexts)
    prompt = create_prompt(query, context_str)
    response = invoke_bedrock(prompt)
    return response



def main():
    parser = argparse.ArgumentParser(description="CLI for querying pinecone index data")
    parser.add_argument("action", choices=["embed", "search", "invoke"], help="Action to perform: 'embed' generate the vector embeddings, 'search' to perform semantic search, 'invoke' to implement RAG")
    parser.add_argument("query", help="Query to be used for embedding, search, or RAG")
    args = parser.parse_args()
    query = args.query

    bedrock = create_bedrock_connection()
    pc = create_pinecone_connection()

    if args.action == "embed":
        embed(query, bedrock)
    elif args.action == "search":
        search(query, bedrock, pc)
    elif args.action == "invoke":
        invoke(query, bedrock, pc)

if __name__ == "__main__":
    main()