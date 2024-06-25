from fastapi import FastAPI, Request
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
from fastapi.staticfiles import StaticFiles

app = FastAPI()

load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

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
        qry_resp = ''.join(output)
        return qry_resp
                
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


@app.post("/submit-question")
async def invoke(request: Request):
    body = await request.json()
    query = body['question']
    bedrock = create_bedrock_connection()
    pc = create_pinecone_connection()
    index = pc.Index(PINECONE_INDEX_NAME)

    query_embedding = titan_text_embeddings(query, bedrock)
    start_time = time.time()
    search_res = index.query(vector=query_embedding, top_k=10, namespace=PINECONE_NAMESPACE,include_metadata=True)
    end_time = time.time()
    print(f"Pinecone query execution time: {(end_time - start_time) * 1000} ms")

    contexts = [match.metadata["text"] for match in search_res.matches]
    context_str = construct_context(contexts=contexts)
    llm_prompt = create_prompt(query, context_str)
    response = invoke_bedrock(llm_prompt, bedrock)
    return {"answer": response}

app.mount("/", StaticFiles(directory="static"), name="static")