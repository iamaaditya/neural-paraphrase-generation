# Code to demonstrate Neural Paraphrase Generation. Includes public Datasets.

Disclaimer: This is not an official repository for the paper mentioned below.
            This is only a re-implementation and no gurantees for any results are made.
``` 
    NOTE: Repository size is 700MB, cloning will take some time
```

[Arxiv](https://arxiv.org/abs/1610.03098)

[Semantic Scholar](https://www.semanticscholar.org/paper/Neural-Paraphrase-Generation-with-Stacked-Residual-Prakash-Hasan/0662db8ec063f14507b43e4f93884c0d0e051d68)

<table width="100%">
  <tr>
  <td width="20%"><img src="https://github.com/iamaaditya/iamaaditya.github.io/raw/master/images/residual_lstm.png" /></td>
	  <td width="80%"> Conventional paraphrase generation methods either leverage hand-written rules and thesauri-based alignments, or use statistical machine learning principles. To the best of our knowledge, this work is the first to explore deep learning models for paraphrase generation. Our primary contribution is a stacked residual LSTM network, where we add residual connections between LSTM layers. This allows for efficient training of deep LSTMs. We evaluate our model and other state-of-the-art deep learning models on three different datasets: PPDB, WikiAnswers and MSCOCO. Evaluation results demonstrate that our model outperforms sequence to sequence, attention-based and bi- directional LSTM models on BLEU, METEOR, TER and an embedding-based sentence similarity metric. </td>
  </tr>
</table>


## Samples

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
```
  -h, --help            : show this help message and exit
  --num_units           : Number of units in a LSTM cell
  --embed_dim           : Size of the embedding vector
  --learning_rate       : learning rate for the optimizer
  --optimizer           : Name of the train source file
  --batch_size          : random seed for training sampling
  --print_every         : print records every n iteration
  --iterations          : number of iterations to train
  --model_dir           : Directory where to save the model
  --input_max_length    : Max length of input sequence to use
  --output_max_length   : Max length of output sequence to use
  --use_residual_lstm   : To use the residual connection with the residual LSTM
  --nouse_residual_lstm : To not use residual LSTM (default)
  --input_filename      : Name of the train source file
  --output_filename     : Name of the train target file
  --vocab_filename      : Name of the vocab file
 ```

## Data

All the data is in public domain and collected from various sources. Please see our paper for details about sources.

### MSCOCO
### PPDB
### Wikianswers
Data is very large and I split them into multiple files. Consider them joining before training. 
Also quality of paraphrases is not that great. Results will be mediocre.

### Snomed (Clinical)

## TODO
- [x] Add data
- [x] Add pretrained models (checkpoints)
- [ ] Add hyper-parameter information for reproducibility

## Pretrained model

- Extract checkpoints.tar.gz
- It was trained in MSCOCO, and gives decent results

```
    ****source == a guy on a bike next to a <UNK>
    ****target == a bicyclist passing a red commuter bus at a stop on a city <UNK>
    ****predict == a man riding a bike on a city <UNK>
    
    ****source == a large giraffe is standing behind a <UNK>
    ****target == a giraffe stands behind a metal chain link <UNK>
    ****predict == a giraffe standing in a field with a <UNK>

    ****source == a small baby giraffe is head butting a wooden post in an <UNK>
    ****target == a couple of giraffe standing next to each <UNK>
    ****predict == a baby giraffe standing next to a <UNK>

    ****source == a cat sitting on a laptop in the <UNK>
    ****target == a cat sits on the keyboard of a <UNK>
    ****predict == a cat sitting on a laptop computer on a <UNK>

    ****source == a person standing near to a train passing <UNK>
    ****target == a railroad train driving down a city <UNK>
    ****predict == a person standing next to a train on a <UNK>

    ****source == a small cat sits with its owner on a <UNK>
    ****target == a cat is lying in a persons <UNK>
    ****predict == a cat is sitting on top of a <UNK>

    ****source == a white transport truck driving down a <UNK>
    ****target == a toy model of a town and highway with <UNK>
    ****predict == a white truck driving down a road next to a <UNK>

    ****source == a large truck and cars on a city <UNK>
    ****target == a tractor trailer leaves a small trail of smoke as it leaves a parking <UNK>
    ****predict == a large truck is parked on the side of a <UNK>
```


## Citation

```
@inproceedings{Prakash2016NeuralPG,
  title={Neural Paraphrase Generation with Stacked Residual LSTM Networks},
  author={Aaditya Prakash and Sadid A. Hasan and Kathy Lee and Vivek Datla and Ashequl Qadir and Joey Liu and Oladimeji Farri},
  booktitle={COLING},
  year={2016}
}
```

