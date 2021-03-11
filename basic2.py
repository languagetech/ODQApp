# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 02:45:03 2021

@author: risha
"""

from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


def find_answer(text: str, question: str):

    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="tf")
    input_ids = inputs["input_ids"].numpy()[0]
    
    #text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    scores = model(inputs)
    answer_start_scores, answer_end_scores = scores[0], scores[1]
        
    answer_start = tf.argmax(
        answer_start_scores, axis=1
    ).numpy()[0]  # Get the most likely beginning of answer with the argmax of the score
    answer_end = (
        tf.argmax(answer_end_scores, axis=1) + 1
    ).numpy()[0]  # Get the most likely end of answer with the argmax of the score
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    
    
    return question, answer

text_ = r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""

question_ = "How many pretrained models are available in Transformers?"
    
#question, answer = find_answer(text_, question_)

#print(f"Question: {question}")
#print(f"Answer: {answer}\n")




from fastapi import FastAPI, Query 
app = FastAPI()

@app.get("/answer/")
async def handle_answer(     
    text: str = Query(
        None,
        alias="t",
        title="Context",
        description="Context for which the is being asked",
        min_length=1,
        #max_length=model.config.max_position_embeddings,
    ),
    
    question: str = Query(
        None,
        alias="q",
        title="Query string",
        description="Query string for the items to search in the database that have a good match",
        min_length=1,
        max_length=model.config.max_position_embeddings,
    )
    ):
    question, answer = find_answer(text, question)
    return {question: answer}



