import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import argparse
import time
import requests
import random
from datetime import date
import os
from pinecone import Pinecone
from transformers import AutoModel, AutoTokenizer
from pinecone import Pinecone
import pandas as pd
import itertools

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")
DATA_DIR = os.path.join(os.path.dirname(__file__), "./jsonl")

# Commented out some sections to reduce the scrape time
# news_sections = ["us", "world", "politics", "business", "health", "entertainment", "style", "travel", "sports"]
news_sections = ["world", "politics", "business"]

def get_article_urls(section):
    articles = []
    response = requests.get(f"https://www.cnn.com/{section}")

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.select('a[data-link-type="article"]')
    site_prefix = "http://cnn.com"
    for link in links:
        articles.append(f"{site_prefix}{link['href']}")

    articles_no_duplicates = list(set(articles))

    return list(articles_no_duplicates)

def get_article_details(urls):
    details = []
    #for url in urls:
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            script_tag = soup.find('script', {'type': 'application/ld+json'})
            data = json.loads(script_tag.string)
            text = data['articleBody']
            details.append({"url": url, "text": text, "scrape_date": date.today().strftime("%m/%d/%Y")})
            print(f"Web scraped article from {url}")
            time.sleep(random.uniform(.1, 1))  
        except Exception as e:
            print(f"Web scraped article from {url}: {e}")
    print(f"Web scraped {len(details)} articles")
    return details

def create_jsonl_file(section, article_details):
    with open(f'jsonl/cnn_articles_{section}.jsonl', 'w') as f:
        for article_detail in article_details:
            for text_embedding in generate_embeddings_from_text(article_detail['text']):
                doc_id = get_article_id(article_detail['url'])
                jsonl_element = {}
                jsonl_element['id'] = f"doc-{doc_id}#chunk{text_embedding['chunk_id']}"
                jsonl_element['values'] = text_embedding['embedding']
                jsonl_element['metadata'] = {"text": text_embedding['text'], 
                                             "scrape_date": article_detail['scrape_date'], 
                                             "section": section, 
                                             "source": article_detail['url']}
               
                f.write(json.dumps(jsonl_element) + '\n')
            print(f"Wrote article as doc_id: {doc_id} to jsonl file")
    print(f"Wrote {len(article_details)} articles to data directory")

def generate_embeddings_from_text(text):
    
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    chunk_size = 512
    overlap = 50
    text_embeddings = []
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
    for chunk in chunks:
        tokens = tokenizer(chunk, return_tensors='pt', padding=True)
        outputs = model(**tokens)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        
        text_embeddings.append({'text':chunk,
                                'chunk_id':chunks.index(chunk),
                                'embedding':embedding })
    return text_embeddings

def scrape():
    for section in news_sections:
        urls = get_article_urls(section)
        article_details = get_article_details(urls)
        create_jsonl_file(section, article_details)

def upsert():
    pc = Pinecone(api_key=API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, 'r') as file:
                df = pd.read_json(file, lines=True)
                index.upsert_from_dataframe(df, namespace=PINECONE_NAMESPACE)

def print_test_vectors():
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, 'r') as file:
                for line in itertools.islice(file, 3):
                    data = json.loads(line)
                    print(f'----- TEST EMBEDDING -----')
                    values = str(data["values"]).replace('[', '').replace(']', '')
                    print(f'{values}\n\n')
                    print(f'----- TEST EMBEDDING METADATA -----')
                    print(f'{data["metadata"]}\n\n')

def delete_data():
    pc = Pinecone(api_key=API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
    print(f"Deleted all vectors in index: {PINECONE_INDEX_NAME} for namespace: {PINECONE_NAMESPACE}")

def get_article_id(url):
    s = url
    s = s[15:]
    s = s.replace("/", "_")
    s = s[:-11]
    return s

def main():
    parser = argparse.ArgumentParser(description="CLI for upserting and deleted pinecone index data")
    parser.add_argument("action", choices=["scrape", "upsert", "delete", "print"], help="Action to perform: 'scrape' to scrape data from base url, 'upsert' to insert or update data, 'delete' to delete all data in namespace")
    args = parser.parse_args()

    if args.action == "scrape":
        scrape()
    elif args.action == "upsert":
        upsert()
    elif args.action == "print":
        print_test_vectors()
    elif args.action == "delete":
        delete_data()

if __name__ == "__main__":
    main()