# Automatic Speech Recognition

## Introduction

An **automatic speech recognition** system should be able to transcribe a given speech utterance to its corresponding transcript, end-to-end. We are provided with the utterances and their corresponding transcript. We can achieve this by using a combination of Recurrent Neural Networks (RNNs) / Convolutional Neural Networks (CNNs) and Dense Networks to design a system for speech to text transcription.

## Design

There are many ways to approach this problem. This project makes use of an **attention** based system. Attention Mechanisms are widely used for various applications these days. More often than not, speech tasks can also be extended to images. Specifically this repo implements a variation of **Listen, Attend and Spell**.

### Listen, Attend and Spell

The idea is to learn all components of a speech recogniser jointly. The paper describes an encoder-decoder approach, called Listener and Speller respectively.

The **Listener** consists of a **Pyramidal Bi-LSTM Network** structure that takes in the given utterances and compresses it to produce high-level representations for the Speller network.

The **Speller** takes in the high-level feature output from the Listener network and uses it to compute a probability distribution over sequences of characters using the **attention mechanism**.

Attention intuitively can be understood as trying to learn a mapping from a word vector to some areas of the utterance map. The Listener produces a high-level representation of the given utterance and the Speller uses parts of the representation (produced from the Listener) to predict the next word in the sequence.

### Variation to LAS

The LAS model only uses a single projection from the Listener network. However, we can instead take two projections and use them as an Attention Key and an Attention Value.

The encoder network in this case produces two outputs, an attention **value** and a **key** and the decoder network over the transcripts will produce an attention query. The dot product between that query and the key is called the **energy** of the attention.

Subsequently, we feed that energy into a Softmax, and use that Softmax distribution as a mask to take a weighted sum from the attention value, that is, apply the attention mask on the values from the encoder. This masked value is called the attention **context**, which is fed back into the transcript network.

### Variable Length Inputs

The transcripts as well as the utterances are of variable length. In order to deal with this problem, we use the built-in pack padded sequence and pad packed sequence APIs from PyTorch. This will pack variable length inputs into a combined tensor input which can be fed into the Encoder.

### Listener/Encoder

The encoder is the part that runs over the utterances to produce attention values and keys. Here we have a batch of utterances and use a layer of Bi-LSTMs to obtain the features. Subsequently we perform a pooling like operation by concatenating outputs. We do this three times as mentioned in the paper and lastly project the final layer output into an attention key and value pair.

### Speller/Decoder

The decoder is an LSTM that takes character[t] as input and produces character[t+1] as output on each time-step. The decoder also receives additional information through the attention context mechanism. As a consequence, we cannot use the LSTM implementation in PyTorch directly, and we
instead have to use LSTMCell to run each time-step in a for loop.

### Teacher Forcing

One problem we encounter in this setting is the difference of training time and evaluation time: at test time we pass in the generated characters from our model (to predict the output at t+1), when our network is used to having perfect labels passed in during training. One way to help our network be better at accounting for this noise is to actually pass in the generated characters during training, rather than the true characters, with some probability. This is known as teacher forcing.

## Dataset and Preprocessing

The Wall Street Journal (WSJ) dataset was used for this work. It contains the raw text. We can either use character-based or word-based model.

Word-based models wont have incorrect spelling and are very quick in training because the sample size decreases drastically. The problem is, it cannot predict rare words.

Character-based models are known to be able to predict some really rare words but at the same time they are slow to train because the model needs to predict character by character.

This repo implements the character-based model. Hence we need to preprocess the data to split the raw text (sentences) into characters and subsequently each character is mapped to a unique integer (refer to VOCAB in dataset.py).

Each transcript/utterance is a separate sample that is a variable length. In order to predict all characters, we need a start and end character added to our vocabulary. We can make them both the same number, like 0, to make things easier.

For example, if the utterance is hello, then:
- inputs=[start]hello
- outputs=hello[end]

Refer to <eos> in VOCAB list in dataset.py.

## Evaluation

Performance is evaluated using CER - character error rate (edit distance).

## Results

The given model achieves **CER of 10.63** on WSJ dataset.

## References

- **Listen, Attend and Spell**: https://arxiv.org/pdf/1508.01211.pdf
