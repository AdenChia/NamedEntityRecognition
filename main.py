from transformers import pipeline
from argparse import ArgumentParser

model_name = './restaurant-model'
nlp = pipeline(task="ner", model=model_name, tokenizer=model_name, framework="pt",grouped_entities=True)

parser = ArgumentParser()
parser.add_argument('--sentence', type=str, required=True)
args = parser.parse_args()

sequence = args.sentence

result = nlp(sequence)

print(result)