# GPT-NeoX-20B

## Model Description

GPT-NeoX-20B is an autoregressive transformer language model trained using [GPT-NeoX](https://github.com/EleutherAI/gpt-neox). "GPT-NeoX" refers to the aforementioned framework, while "20B" represents the number of trainable parameters.

**Hyperparameter**|**Value**
:-----:|:-----:
Num. parameters|20,556,201,984
Num. layers|44
D\_model|6,144
D\_ff|24,576
Num. Heads|64
Context Size|2,048
Vocab Size|50257/50432*
Positional Encoding|[Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864)
Rotary Dimensions|25%
Tensor Parallel Size|2
Pipeline Parallel Size|4

\* The embedding matrix is padded up to 50432 in order to be divisible by 128, but only 50257 entries are used by the tokenizer.

The model consists of 44 layers with a model dimension of 6144, and a feedforward dimension of 24,576. The model dimension is split into 64 heads, each with a dimension of 96. Rotary Position Embedding is applied to the first 24 dimensions of each head. The model is trained with the same vocabulary size as in GPT-2/GPT-3, but with a new tokenizer trained on [the Pile](https://pile.eleuther.ai/), our curated pretraining dataset (described below).


## Training data

GPT-NeoX-20B was trained on [the Pile](https://pile.eleuther.ai/), a large-scale curated dataset created by EleutherAI.

## Training procedure

GPT-NeoX-20B was trained for 470 billion tokens over 150,000 steps on 96 40GB A100 GPUs for around three months. It was trained as an autoregressive language model, using cross-entropy loss to maximize the likelihood of predicting the next token correctly.

## Intended Use and Limitations

GPT-NeoX-20B learns an inner representation of the English language that can be used to extract features useful for downstream tasks. The model is best at what it was pretrained for however, which is generating text from a prompt.

Due to the generality of the pretraining set, it has acquired the ability to generate completions across a wide range of tasks - from programming to fiction writing.

## Limitations and Biases

The core functionality of GPT-NeoX-20B is taking a string of text and predicting the next token. While language models are widely used for tasks other than this, there are a lot of unknowns with this work. When prompting GPT-NeoX-20B it is important to remember that the statistically most likely next token is often not the token that produces the most "accurate" text. Never depend upon GPT-NeoX-20B to produce factually accurate output.

GPT-NeoX-20B was trained on [the Pile](https://pile.eleuther.ai/), a dataset known to contain profanity, lewd, and otherwise abrasive language. Depending upon use case GPT-NeoX-20B may produce socially unacceptable text. See Sections 5 and 6 of [the Pile paper](https://arxiv.org/abs/2101.00027), or [the Pile Datasheet](https://arxiv.org/abs/2201.07311) for a more detailed analysis of the biases in the Pile

As with all language models, it is hard to predict in advance how GPT-NeoX-20B will respond to particular prompts and offensive content may occur without warning. We recommend having a human curate or filter the outputs before releasing them, both to censor undesirable content and to improve the quality of the results.
## Zero Shot Evaluation results

<figure>

| Model            | Public | Training FLOPs | LAMBADA PPL ↓ | LAMBADA Acc ↑ | Winogrande ↑ | Hellaswag ↑ | PIQA ↑    | SciQ      | Hendrycks Humanities | Hendrycks Math | Hendrycks STEM | Hendrycks Social Science | Hendrycks Other | Dataset Size (GB) |
|------------------|--------|----------------|---------------|---------------|--------------|-------------|-----------|-----------|----------------------|----------------|----------------|--------------------------|-----------------|-------------------|
| Random Chance    | ✓      | 0              | ~a lot        | ~0%           | 50%          | 25%         | 25%       | 25%       | 25%                  | 0%             | 25%            | 25%                      | 25%             | 0                 |
| GPT-3 Ada‡       | ✗      | -----          | 9.95          | 51.6%         | 52.9%        | 43.4%       | 70.5%     | 84.3%     | 25.2%                | 0.32%          | 24.2%          | 26.3%                    | 26.4%           | -----             |
| GPT-2 1.5B       | ✓      | -----          | 10.63         | 51.21%        | 59.4%        | 50.9%       | 70.8%     | -----     | -----                | -----          | -----          | -----                    | -----           | 40                |
| GPT-Neo 1.3B‡    | ✓      | 3.0e21         | 7.50          | 57.2%         | 55.0%        | 48.9%       | 71.1%     | -----     | -----                | -----          | -----          | -----                    | -----           | 825               |
| Megatron-2.5B*   | ✗      | 2.4e21         | -----         | 61.7%         | -----        | -----       | -----     | -----     | -----                | -----          | -----          | -----                    | -----           | 174               |
| GPT-Neo 2.7B‡    | ✓      | 6.8e21         | 5.63          | 62.2%         | 56.5%        | 55.8%       | 73.0%     | -----     | -----                | -----          | -----          | -----                    | -----           | 825               |
| GPT-3 1.3B*‡     | ✗      | 2.4e21         | 5.44          | 63.6%         | 58.7%        | 54.7%       | 75.1%     | -----     | -----                | -----          | -----          | -----                    | -----           | ~800              |
| GPT-3 Babbage‡   | ✗      | -----          | 5.58          | 62.4%         | 59.0%        | 54.5%       | 75.5%     | 86.6%     | 26.5%                | 0.38%          | 25.8%          | 26.1%                    | 26.8%           | -----             |
| Megatron-8.3B*   | ✗      | 7.8e21         | -----         | 66.5%         | -----        | -----       | -----     | -----     | -----                | -----          | -----          | -----                    | -----           | 174               |
| GPT-3 2.7B*‡     | ✗      | 4.8e21         | 4.60          | 67.1%         | 62.3%        | 62.8%       | 75.6%     | -----     | -----                | -----          | -----          | -----                    | -----           | ~800              |
| Megatron-11B†    | ✓      | 1.0e22         | -----         | -----         | -----        | -----       | -----     | -----     | -----                | -----          | -----          | -----                    | -----           | 161               |
| GPT-J 6B         | ✓      | 1.5e22         | 3.99          | 69.7%         | 65.3%        | 66.1%       | 76.5%     |   91.5%        | 26.8%                | 0.84%          | 25.6%          | 28.4%                    | 28.1%           | 825               |
| GPT-3 6.7B*‡     | ✗      | 1.2e22         | 4.00          | 70.3%         | 64.5%        | 67.4%       | 78.0%     | -----     | -----                | -----          | -----          | -----                    | -----           | ~800              |
| GPT-3 Curie‡     | ✗      | -----          | 4.00          | 69.3%         | 65.6%        | 68.5%       | 77.9%     | 91.8%     | 26.0%                | 0.42%          | 24.1%          | 27.7%                    | 28.2%           | -----             |
| GPT-3 13B*‡      | ✗      | 2.3e22         | 3.56          | 72.5%         | 67.9%        | 70.9%       | 78.5%     | -----     | -----                | -----          | -----          | -----                    | -----           | ~800              |
| **GPT-NeoX 20B** | **✓**  | **5.7e+22**    | **3.66**      | **72.0%**     | **66.2%**    | **71.3%**   | **78.6%** | **92.8%** | **27.1%**            | **1.1%**       | **27.8%**      | **29.8%**                | **29.6%**       | **825**           |
| GPT-3 175B*‡     | ✗      | 3.1e23         | 3.00          | 76.2%         | 70.2%        | 78.9%       | 81.0%     | -----     | -----                | -----          | -----          | -----                    | -----           | ~800              |
| GPT-3 Davinci‡   | ✗      | -----          | 3.0           | 75%           | 72%          | 78%         | 80%       | 95.0%     | 29.18%               | 0.72%          | 29.1%          | 34.7%                    | 37.8%           | -----             |
<figcaption><p>Models roughly sorted by performance, or by FLOPs if not available.</p>

<p><strong>&ast;</strong> Evaluation numbers reported by their respective authors. All other numbers are provided by
running <a href="https://github.com/EleutherAI/lm-evaluation-harness/"><code>lm-evaluation-harness</code></a> either with released
weights or with API access. Due to subtle implementation differences as well as different zero shot task framing, these
might not be directly comparable. See <a href="https://blog.eleuther.ai/gpt3-model-sizes/">this blog post</a> for more
details.</p>

<p><strong>†</strong> Megatron-11B provides no comparable metrics, and several implementations using the released weights do not
reproduce the generation quality and evaluations. (see <a href="https://github.com/huggingface/transformers/pull/10301">1</a>
<a href="https://github.com/pytorch/fairseq/issues/2358">2</a> <a href="https://github.com/pytorch/fairseq/issues/2719">3</a>)
Thus, evaluation was not attempted.</p>

<p><strong>‡</strong> These models have been trained with data which contains possible test set contamination. The OpenAI GPT-3 models
failed to deduplicate training data for certain test sets, while the GPT-Neo models as well as this one is
trained on the Pile, which has not been deduplicated against any test sets.</p></figcaption></figure>

## Citation and Related Information

To cite this model:
```
@article{gpt-neox-20b,
  title={GPT-NeoX-20B: An Open-Source Autoregressive Language Model},
  author={Black, Sid and Biderman, Stella and Hallahan, Eric and Anthony, Quentin and Gao, Leo and Golding, Laurence and He, Horace and Leahy, Connor and McDonnel, Kyle and Phang, Jason and Pieler, Michael and USVSN Sai, Prashanth and Purohit, Shivanshu and Reynolds, Laria and Tow, Jonathan and Wang, Ben and Weinbach, Samuel},
  journal={arXiv preprint},
  year={2022}
}
```

To cite GPT-NeoX, the training codebase:
```
@software{gpt-neox,
  author = {Andonian, Alex and Anthony, Quentin and Biderman, Stella and Black, Sid and Gali, Preetham and Gao, Leo and Hallahan, Eric and Levy-Kramer, Josh and Leahy, Connor and Nestler, Lucas and Parker, Kip and Pieler, Michael and Purohit, Shivanshu and Songz, Tri and Wang, Phil and Weinbach, Samuel},
  title = {{GPT-NeoX}: Large Scale Autoregressive Language Modeling in PyTorch},
  url = {http://github.com/eleutherai/gpt-neox},
  year = {2022}
}
```