# -*- coding: utf-8 -*-

from flask import Flask, request, render_template
import torch
import os
#from transformers import BertForQuestionAnswering
#from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
app = Flask(__name__)

name = "mrm8488/bert-small-finetuned-squadv2"
tokenizer = AutoTokenizer.from_pretrained(name,)
model = AutoModelForQuestionAnswering.from_pretrained(name)




def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''

    #model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    #tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
   # print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                                     token_type_ids=torch.tensor(
                                         [segment_ids]))  # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    return answer


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/bert', methods=['Post'])
def bert():
    if request.method == 'POST':
        question = request.form['Question']
        answer_text=request.form['Abstract Paragraph']
        if question != '' and answer_text != '':
            predict = answer_question(question, answer_text)

        elif question == '':
            predict='Please Enter your Question'
        elif answer_text == '':
            predict='Please Enter your Abstract Also'

    return render_template('index.html', prediction=predict,question=question)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
