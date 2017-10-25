# Neural Paraphrase Generation using Stacked Residual LSTM
[Arxiv](https://arxiv.org/abs/1610.03098)
[Semantic Scholar](https://www.semanticscholar.org/paper/Neural-Paraphrase-Generation-with-Stacked-Residual-Prakash-Hasan/0662db8ec063f14507b43e4f93884c0d0e051d68)

# CODE + DATA

##Abstract

In this paper, we propose a novel neural approach for paraphrase generation. 
Conventional para- phrase generation methods either leverage hand-written rules 
and thesauri-based alignments, or use statistical machine learning principles. 
To the best of our knowledge, this work is the first to explore deep learning 
models for paraphrase generation. Our primary contribution is a stacked residual 
LSTM network, where we add residual connections between LSTM layers. This allows for 
efficient training of deep LSTMs. We evaluate our model and other state-of-the-art 
deep learning models on three different datasets: PPDB, WikiAnswers and MSCOCO. 
Evaluation results demonstrate that our model outperforms sequence to sequence, 
attention-based and bi- directional LSTM models on BLEU, METEOR, TER and 
an embedding-based sentence similarity metric.

![Neural Paraphrase Generation](https://github.com/iamaaditya/iamaaditya.github.io/raw/master/images/residual_lstm.png)

##Samples

![Samples](https://github.com/iamaaditya/iamaaditya.github.io/raw/master/images/paraphrase_samples.png)


## Quick start

## To Use Residual LSTM
```bash
	python train.py --use_residual_lstm
```
## To Use standard LSTM
```bash
	python train.py --nouse_residual_lstm
```

## Detailed Usage
#
usage: train.py [-h] [--num_units NUM_UNITS] [--embed_dim EMBED_DIM]
                [--learning_rate LEARNING_RATE] [--optimizer OPTIMIZER]
                [--batch_size BATCH_SIZE] [--print_every PRINT_EVERY]
                [--iterations ITERATIONS] [--model_dir MODEL_DIR]
                [--input_max_length INPUT_MAX_LENGTH]
                [--output_max_length OUTPUT_MAX_LENGTH]
                [--use_residual_lstm [USE_RESIDUAL_LSTM]]
                [--nouse_residual_lstm] [--input_filename INPUT_FILENAME]
                [--output_filename OUTPUT_FILENAME]
                [--vocab_filename VOCAB_FILENAME]

optional arguments:
  -h, --help            show this help message and exit
  --num_units NUM_UNITS
                        Number of units in a LSTM cell
  --embed_dim EMBED_DIM
                        Size of the embedding vector
  --learning_rate LEARNING_RATE
                        learning rate for the optimizer
  --optimizer OPTIMIZER
                        Name of the train source file
  --batch_size BATCH_SIZE
                        random seed for training sampling
  --print_every PRINT_EVERY
                        print records every n iteration
  --iterations ITERATIONS
                        number of iterations to train
  --model_dir MODEL_DIR
                        Directory where to save the model
  --input_max_length INPUT_MAX_LENGTH
                        Max length of input sequence to use
  --output_max_length OUTPUT_MAX_LENGTH
                        Max length of output sequence to use
  --use_residual_lstm [USE_RESIDUAL_LSTM]
                        To use the residual connection with the residual LSTM
  --nouse_residual_lstm
  --input_filename INPUT_FILENAME
                        Name of the train source file
  --output_filename OUTPUT_FILENAME
                        Name of the train target file
  --vocab_filename VOCAB_FILENAME
                        Name of the vocab file




