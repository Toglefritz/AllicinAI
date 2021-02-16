# gpt-2-simple

![gen_demo](docs/gen_demo.png)

This repository exists to make working with [OpenAI](https://openai.com)'s [GPT-2 text generation model](https://openai.com/blog/better-language-models/) faster and easier. The repository contains a number of Python scrips designed to streamline the process of finetuning the GPT-2 model so it can generate custom text putput. Additionally, this package allows prefixes to force the text to start with a given phrase.

This package incorporates and makes minimal low-level changes to:

* Forked from: https://github.com/minimaxir/gpt-2-simple:
	* Model management from OpenAI's [official GPT-2 repo](https://github.com/openai/gpt-2) (MIT License)
	* Model finetuning from Neil Shepperd's [fork](https://github.com/nshepperd/gpt-2) of GPT-2 (MIT License)
	* Text generation output management from [textgenrnn](https://github.com/minimaxir/textgenrnn) (MIT License / also created by me)

For finetuning, it is **strongly** recommended to use a GPU, otherwise the process will be much, much slower on a CPU. This repository is designed around implementation in Google Colab. Colab provides cloud-hosted GPU environments that run Python code. The GPUs and TPUs provided by Google Colab would likely be too costly for many users. If you need to ensure access to a high-end GPU or if you need longer run times, Google Colab offers a paid, Colab Pro, plan that includes both these perks, among others.

You can use gpt-2-simple to retrain a model using a GPU in [this Colaboratory notebook](https://colab.research.google.com/drive/1kaZtMprzST3ztVgbO-8AhlKN46OjGb1G).

## Install

This repository can be cloned into your Google Drive account via Colab:

```# Connect to Google Drive
from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
Path("gpt2finetune").mkdir(parents=True, exist_ok=True)
%cd gpt2finetune/

# Clone the AllicinAI GPT-2 repository
!git clone https://github.com/Toglefritz/AllicinAI.git
%cd gpt-2
```

You will also need to install the corresponding TensorFlow for your system (e.g. `tensorflow` or `tensorflow-gpu`). **TensorFlow 2.0 is currently not supported** and the package will throw an assertion if loaded, so TensorFlow 1.15 is recommended:

```%tensorflow_version 1.x
# !pip -q install tensorflow==1.15 && pip -q install tensorflow-gpu==1.15
# !pip -q install 'tensorflow-estimator<1.15.0rc0,>=1.14.0rc0' --force-reinstall
import tensorflow
```

You can check the TensorFlow version to ensure you successfully installed 1.15:

``` print(tensorflow.__version__) ```

## Usage

An example for downloading the model to the local system, finetuning it on a dataset. and generating some text.

Warning: the pretrained 124M model, and thus any finetuned model, is 500 MB! (the pretrained 355M model is 1.5 GB)

```python
import gpt_2_simple as gpt2
import os
import requests

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


file_name = "shakespeare.txt"
if not os.path.isfile(file_name):
	url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	data = requests.get(url)
	
	with open(file_name, 'w') as f:
		f.write(data.text)
    

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              model_name=model_name,
              steps=1000)   # steps is max number of training steps

gpt2.generate(sess)
```

The generated model checkpoints are by default in `/checkpoint/run1`. If you want to load a model from that folder and generate text from it:

```python
import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

gpt2.generate(sess)
```

As with textgenrnn, you can generate and save text for later use (e.g. an API or a bot) by using the `return_as_list` parameter.

```python
single_text = gpt2.generate(sess, return_as_list=True)[0]
print(single_text)
```

You can pass a `run_name` parameter to `finetune` and `load_gpt2` if you want to store/load multiple models in a `checkpoint` folder.

There is also a command-line interface for both finetuning and generation with strong defaults for just running on a Cloud VM w/ GPU. For finetuning (which will also download the model if not present):

```shell
gpt_2_simple finetune shakespeare.txt
```

And for generation, which generates texts to files in a `gen` folder:

```shell
gpt_2_simple generate
```

Most of the same parameters available in the functions are available as CLI arguments, e.g.:

```shell
gpt_2_simple generate --temperature 1.0 --nsamples 20 --batch_size 20 --length 50 --prefix "<|startoftext|>" --truncate "<|endoftext|>" --include_prefix False --nfiles 5
```

See below to see what some of the CLI arguments do.

NB: *Restart the Python session first* if you want to finetune on another dataset or load another model.

## Interactive Apps Using gpt-2-simple

* [gpt2-small](https://minimaxir.com/apps/gpt2-small/) — App using the default GPT-2 124M pretrained model
* [gpt2-reddit](https://minimaxir.com/apps/gpt2-reddit/) — App to generate Reddit titles based on a specified subreddit and/or keyword(s)
* [gpt2-mtg](https://minimaxir.com/apps/gpt2-mtg/) — App to generate Magic: The Gathering cards

## Text Generation Examples Using gpt-2-simple

* [ResetEra](https://www.resetera.com/threads/i-trained-an-ai-on-thousands-of-resetera-thread-conversations-and-it-created-hot-gaming-shitposts.112167/) — Generated video game forum discussions ([GitHub w/ dumps](https://github.com/minimaxir/resetera-gpt-2))
* [/r/legaladvice](https://www.reddit.com/r/legaladviceofftopic/comments/bfqf22/i_trained_a_moreadvanced_ai_on_rlegaladvice/) — Title generation ([GitHub w/ dumps](https://github.com/minimaxir/legaladvice-gpt2))
* [Hacker News](https://github.com/minimaxir/hacker-news-gpt-2) — Tens of thousands of generated Hacker News submission titles

## Maintainer/Creator

Toglefritz ([@Toglefritz](https://toglefritz.com/))

## License

MIT

## Disclaimer

This repo has no affiliation or relationship with OpenAI.
