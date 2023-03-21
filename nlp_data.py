import nlpcloud
import pandas as pd
import time
from sklearn.model_selection import train_test_split

notes = pd.read_csv('notes.csv').sample(frac = 0.1)

def nlp_scrape(index):
    global start_index, prompt
    for i in range(index, len(notes)):
        start_index = i
        with open("prompts2.txt", 'r') as f:
            prompt = f.read()
        last_five = f"[Notes]: {'[Notes]: '.join(prompt.split('[Notes]: ')[-4:])}"
        prompt_input = f"{last_five}\n[Notes]: {notes.iloc[i,0]}\n[Summary]:"
        client = nlpcloud.Client("finetuned-gpt-neox-20b", "b1cedee8a0427eb9a16a035faf8586ef46a1d440", gpu=True, lang="en")
        response = client.generation(
            prompt_input,
            min_length=10,
            max_length=50,
            length_no_input=True,
            remove_input=True,
            end_sequence=None,
            top_p=1,
            temperature=0.4,
            top_k=50,
            repetition_penalty=1,
            length_penalty=1,
            do_sample=True,
            early_stopping=False,
            num_beams=1,
            no_repeat_ngram_size=0,
            num_return_sequences=1,
            bad_words=None,
            remove_end_sequence=False
        )
        output = f"\n[Notes]: {notes.iloc[i,0]}\n[Summary]: {response['generated_text']}"
        with open('prompts2.txt', 'a') as f:
            f.write(f"{output}")
        time.sleep(30)


start_index = 18
start = 1
while start == 1:
    try:
        nlp_scrape(start_index)
        start = 0
    except Exception as e:
        print(e)
        time.sleep(3600)

# 1b3fdd3f6f0bbdb70675d4402ca55767d41588ed 
# 98026b5681e4bd90a723af8dae86261465021fda
# b1cedee8a0427eb9a16a035faf8586ef46a1d440

# Convert dataset to json
import json

with open("prompts2.txt", 'r') as f:
    prompt = f.read()

prompt_list = prompt.split('[Notes]: ')
prompt_list.pop(0)
output = {}
for i in range(len(prompt_list)):
    row = prompt_list[i].split('[Summary]: ') 
    output['prompt'] = row[0]
    output['completion'] = row[1]
    with open('dataset.jsonl', 'a') as f:
        json.dump(output, f)
        f.write('\n')

from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, GPTJForCausalLM, AutoModelForCausalLM
from datasets import load_dataset
import torch
dataset = load_dataset("json", data_files="dataset.jsonl")

### Creating Validation Set ###
dataset = dataset["train"].train_test_split(0.25, seed=42)
split_dataset = dataset["train"].train_test_split(0.2, seed=42)
split_dataset["validation"] = dataset["test"]


## Convert jsonl to json in terminal ##
# cat dataset.jsonl | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/,/g'  | sed 's/n/,/' | sed 's/^/[/'| sed 's/$/]/' > dataset1.json

# with open('dataset.jsonl', 'r') as f:
#     json_list = list(f)

# dataset = load_dataset("json", data_files="dataset.jsonl")

### GPT-J6B ###
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = GPTJForCausalLM.from_pretrained("bryanmildort/gpt-clinical-notes-summarizer", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_8bit=True
)


### GPT-2XL ###
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
# model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

### GPT-Neo20B ###
# model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
# tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

### GPT-JT ###
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1")
max_positions = 2048
for i in range(len(model.transformer.h)):
    model.transformer.h[i].attn.bias[:] = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
        1, 1, max_positions, max_positions
    )

### GPT-Neo2.7B ###
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

### GPT-2XL ###
# tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
# model = AutoModelForCausalLM.from_pretrained("gpt2-xl")

### Pythia2.8B ###
# model = GPTNeoXForCausalLM.from_pretrained(
#   "EleutherAI/pythia-2.8b-deduped",
#   revision="step143000",
#   cache_dir="./pythia-2.8b-deduped/step143000",
# )

# tokenizer = AutoTokenizer.from_pretrained(
#   "EleutherAI/pythia-2.8b-deduped",
#   revision="step143000",
#   cache_dir="./pythia-2.8b-deduped/step143000",
# )

### BioBERT ###
# tokenizer = AutoTokenizer.from_pretrained("bvanaken/CORe-clinical-outcome-biobert-v1")
# model = BertForMaskLM.from_pretrained("bvanaken/CORe-clinical-outcome-biobert-v1", is_decoder=True)

from transformers import MegatronBertModel, BertTokenizer

column_names = dataset["train"].column_names
preprocessing_num_workers = None

def tokenize_function(examples):
    return tokenizer(examples["PARSED"], examples["Hypoglycemia"], padding=True, truncation=True, max_length=512)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=preprocessing_num_workers, remove_columns=column_names, keep_in_memory=True)

model.resize_token_embeddings(len(tokenizer)) ##### THIS IS HOW YOU RESIZE THE MODEL TO MATCH THE TOKENIZER!!!!

# def copy_texts(examples):
#     examples["labels"] = examples["input_ids"].copy()
#     return examples

# tokenized_datasets = tokenized_dataset.map(
#     copy_texts, 
#     batched=True, 
#     num_proc=preprocessing_num_workers
# )

import torch
import random
import numpy as np
seed_val = 23
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


training_args = TrainingArguments(output_dir="test_trainer", save_total_limit = 2, save_strategy="no", load_best_model_at_end=False)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator,
    #eval_dataset=encoded_eval,
    #compute_metrics=compute_metrics,
)

# huggingface-cli lfs-enable-largefiles ./path/to/your/repo

### TO FREE UP MEMORY AND DISK SPACE FROM USING LARGE LM (6B+) ###
import shutil
shutil.rmtree("/home/bryan/.cache") 
del dataset

trainer.train()
trainer.save_model(output_dir='./gpt_jt')
tokenizer.save_vocabulary('./gpt_jt')


############


# encoded_input = tokenizer(dataset, padding=True, return_tensors="pt")

# metric = evaluate.load("accuracy")
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# gsutil -m cp -R gs://notes-nlp/NOTEEVENTS.csv .

from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModel, BertLMHeadModel

# model = AutoModel.from_pretrained("healx/gpt-2-pubmed-medium")
# tokenizer = AutoTokenizer.from_pretrained("healx/gpt-2-pubmed-medium")

# model = GPTNeoXForCausalLM.from_pretrained(
#   "EleutherAI/pythia-2.8b-deduped",
#   revision="step143000",
#   cache_dir="./pythia-2.8b-deduped/step143000",
# )

# tokenizer = AutoTokenizer.from_pretrained(
#   "EleutherAI/pythia-2.8b-deduped",
#   revision="step143000",
#   cache_dir="./pythia-2.8b-deduped/step143000",
# )


prompt = """[Notes]: 26yo female presents for ED f/u after episode of heart palpitations and hand numbness. 2 weeks ago. EKG, CBC, CMP, and cardiac enzymes were all within normal limits. Palpitations started 5 years ago. Described as intense pounding. Increased in the past 3 weeks to 2-3 days per week. With episode also experiences throat tightness, SOB, nausea, hot flashes, and cold and clammy feeling. No changes in skin or hair. Recently lost job as sales associate. No caffeine intake. No alcohol, smoking or drug use. Feels stressed but not overwhelmed. SIGECAPS negative. No worries about work or relationship.
ROS negative except for HPI
PMH none
PSH none
FH none
Ob/gyn 1 sexual partner. Uses condoms. recent STD screen negative. 
Social hx
[Diagnosis]: Follow up for Heart Palpitations
[Notes]: CC: 44 F with irregular periods x 3 years
HPI: 44 yo F without significant medical history presents with irregular menstruation since 3 years ago. They occur 3 weeks to 4 months apart lasting for 2-6 days with light to heavy flow. LMP was 2 months ago. Previously had very regular periods every 28-29 days lasting for 5 days with moderate flow associated with breast tenderness, PMS symptoms, and lower back pain. Additionally 1 week ago she had a few days of n/v that resolved, and she also had an episode last week where she awoke with her sheets drenched in sweat. She also has noticed vaginal dryness for which she takes a lubricant. Has 2 children born vaginally, uncomplicate
ROS: neg for HA, dizziness, constipation, diarrhea, wt loss
PMH: HTN
PSH: none
Meds: HCTZ 12.5 mg daily for hypertension
Allergies: none
Fam Hx: brother with hypertension, mother with osteoarthritis
Soc Hx: no tobacco, rare alc, sex active w/ husband, 2 kids
[Diagnosis]: Menopausal symptoms
[Notes]: HPI: 20 y/o F C/o 1 day headache, doll,8-10/10, costant ,no alliviating whit tynelol, ibuprofen or sleep,, worsering  light, no changes w/ sounds or food, first episode,vomiting , nausea 

PMH: Medical:-; PSH;-; All:-;Medication:bird control pills; Ob/Gyn LMP 2w/a, 28x5 , bird control pills and condom use; FH; DAD: High Cholesterol, MOM : Migrane
SH:Alcohol:-;Smoke:-;Drugs: Marijuna; Sales personal;SxH: active  always condoms use

ROS:-
[Diagnosis]: Severe headache
[Notes]: Angela Tompkins is a 35-year-old female with no significant PMHx presenting with abnormal uterine bleeding. 

-6 mos ago pt noticed periods were less frequent and assocaited with heavier bleeding 

-LMP was 2 mos ago

-Pt states she has to change tampon every few hours where previously it was 3-4 per day

-gained 10 lbs in the last 6 mos which she attributes to eating out more freqeuently 

- normal pap smear 6 mos ago 

-ROS pos for: weight gain, fatigue

-ROS negative for: abdominal pain, pain with intercourse, easy bruising, mucosal bleeding, constipation, heat/cold intolerance, changes in bowel or bladder habbits, hirsutism 

-PMHx: pt was unable to get pregnant in her previous relationship, no meds/allergies/injuries/hospitilizations/surgeries

-family hx: MGM with cervical cancer, maternal aunt with breast cancer 

-social: has 2 adopted children age 3 and 5, sexually active with BF, use condoms for protection, never pregnant
[Diagnosis]: """

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.8,
    max_length=888
)

gen_text = tokenizer.batch_decode(gen_tokens)[0]
gen_text


# import os, torch
# torch.save(model.state_dict(), 'notes_gpt.bin')
# model.config.to_json_file('notes_gpt_config.json')
# tokenizer.save_vocabulary(os.getcwd())

### Neo text generation ###
# inputs = tokenizer(prompt, return_tensors="pt")
# tokens = model.generate(**inputs, min_length=184, max_length=192, temperature=0.5)
# tokenizer.decode(tokens[0])

# gcloud compute tpus tpu-vm create gpt-neox --zone us-central1-f --accelerator-type v2-8 --version tpu-vm-base
# shutil.rmtree("/home/bryan/.cache")

# GPT2 Text gen
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='./my_model1')
set_seed(42)
generator(prompt, min_length=2, max_length=12, temperature=0.5, num_return_sequences=1)

notes_list = list(notes.TEXT)
for i in notes_list:
    with open('mimic_notes.txt', 'a') as f:
        f.write(i)   
        f.write('\n\n##########\n\n')

# du -a | sort -n -r | head -n 50

#### GOOGLE COLAB TERMINAL ####
# !pip install colab-xterm
# %load_ext colabxterm
# %xterm

########### AMAZON EC2 PREP ############
#### Disk Partition ####
# lsblk
# sudo parted /dev/nvme1n1
# mklabel gpt
# unit GB
# mkpart primary ext2 0.0GB 558.8GB
# print
# quit
# sudo mkfs.ext4 /dev/nvme1n1
# sudo mkdir ./store
# sudo vim /etc/fstab # add: /dev/nvme1n1 /home/ec2-user/store ext4 defaults 0 0
# sudo mount ./store
# sudo chmod 777 ./store
# sudo chown -R $USER:$USER ./store

# rm -rf ./anaconda3/envs/amazonei_pytorch_latest_p37
# rm -rf ./anaconda3/envs/amazonei_pytorch_latest_p37/tensorflow2_p310
# rm -rf ./anaconda3/envs/tensorflow2_p310
# rm -rf ./anaconda3/envs/mxnet_p38
# rm -rf ./anaconda3/envs/amazonei_mxnet_p36
# rm -rf ./anaconda3/envs/amazonei_tensorflow2_p36
# rm -rf ./anaconda3/envs/R

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


#############
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, MegatronBertForCausalLM
from transformers import BertModel
if torch.cuda.is_available():   
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained('./MegatronBERT')
model = MegatronBertModel.from_pretrained('./MegatronBERT')

# for i in range(len(notes)):
#     prompt = notes.iloc[i]['prompt']
#     completion = notes.iloc[i]['completion']
#     sentence = f"[Notes]: {prompt}[Summary]: {completion}"
#     encoded_dict = tokenizer.encode_plus(
#                         sentence,                      
#                         #add_special_tokens = True, 
#                         max_length = 512,           
#                         padding = True,
#                         truncation = True,
#                         return_attention_mask = True,   
#                         return_tensors = 'pt',    
#                    )    
#     input_ids.append(encoded_dict['input_ids'])
#     attention_masks.append(encoded_dict['attention_mask'])

dataset = load_dataset("json", data_files="dataset1.json")

def tokenize_function(examples):
    return tokenizer(examples["prompt"], examples["completion"], return_tensors='pt', padding=True, truncation=True, max_length=512)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model.resize_token_embeddings(len(tokenizer))

tokenized_dataset = dataset.map(tokenize_function, batched=True, keep_in_memory=True)

input_ids = []
attention_masks = []
input_tokens = tokenized_dataset['train']['input_ids']
att_tokens = tokenized_dataset['train']['attention_mask']
for i in range(len(tokenized_dataset['train']['input_ids'])):
    input_ids.append(torch.tensor(input_tokens[i]).unsqueeze(dim=0))
    attention_masks.append(torch.tensor(att_tokens[i]).unsqueeze(dim=0))

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

from torch.utils.data import TensorDataset, random_split
from torch.optim import AdamW
dataset = TensorDataset(input_ids, attention_masks)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
batch_size = 32
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

# Model Parameter Observations ##
params = list(model.named_parameters())
print('The BERT model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


print('\n==== First Transformer ====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

from transformers import get_linear_schedule_with_warmup
epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

import random, time, datetime
import numpy as np
seed_val = 26
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
training_stats = []
total_t0 = time.time()

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))    
    return str(datetime.timedelta(seconds=elapsed_rounded))

# For each epoch...
for epoch_i in range(0, epochs):
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()
    # Reset the total loss for this epoch.
    total_train_loss = 0
    model.train()
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        model.zero_grad()        
        # Perform a forward pass (evaluate the model on this training batch).
        # loss, logits = model(b_input_ids, 
        #                      token_type_ids=None, 
        #                      attention_mask=b_input_mask)
        # # Accumulate the training loss over all of the batches
        # total_train_loss += loss.item()
        # Perform a backward pass to calculate the gradients.
        # loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        optimizer.step()
        # Update the learning rate.
        scheduler.step()
    # Calculate the average loss over all of the batches.
    # avg_train_loss = total_train_loss / len(train_dataloader)
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("")
    # print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))        
# Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            # 'Training Loss': avg_train_loss,
            'Training Time': training_time,
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained('./bert_model')
tokenizer.save_pretrained('./bert_model')

output_dir = './gpt-j-6B'
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model=output_dir)
set_seed(23)
generator(prompt, max_length=264, temperature=0.5, num_return_sequences=1)

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2Model.from_pretrained('gpt2-medium')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

max_length = len(output['logits'][0]) + 50

###### Handling Bigger Models for Inference ######
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM


config = AutoConfig.from_pretrained(checkpoint)

# with init_empty_weights():
#     model = AutoModelForCausalLM.from_config(config)

# from accelerate import load_checkpoint_and_dispatch

# model = load_checkpoint_and_dispatch(
#     model, checkpoint, device_map="auto", no_split_module_classes=["GPTJBlock"]
# )

checkpoint = "bryanmildort/gpt-notes-summarizer-demo"
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", offload_folder="offload", offload_state_dict = True, no_split_module_classes=["GPTJBlock"], torch_dtype=torch.float16)

from accelerate import infer_auto_device_map, init_empty_weights
device_map = infer_auto_device_map(model, dtype="float16")

notes_input = tokenizer("Hi my name is ", return_tensors='pt')
output = model(**notes_input)
max_length = len(output['logits'][0]) + 50
input_ids = notes_input.input_ids
gen_tokens = model.generate(input_ids, do_sample=True, temperature = 0.5, max_length=max_length)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
return gen_text.replace(notes, '')

# /dev/nvme1n1             /home/ec2-user/store                  ext4    defaults        0 0

def summarize_function(notes):
    max_length = len(notes.split(' ')) + 50
    return tokenizer.batch_decode(
        (
            model.generate(
                tokenizer(notes, return_tensors="pt").input_ids,
                do_sample=True,
                temperature=0.5,
                max_length=250,
            )
        )
    )
    
def summarize_function(notes):
    notes_input = tokenizer(notes, return_tensors='pt')
    # output = model(**notes_input)
    # max_length = len(output['logits'][0]) + 50
    max_length = len(notes.split(' ')) + 50
    input_ids = notes_input.input_ids
    gen_tokens = model.generate(input_ids, do_sample=True, temperature = 0.5, max_length=max_length)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text.replace(notes, '')


# The task is to generate a short summary of the given clinical notes. Do not repeat the notes verbatim.
# 20 y/o female c/o abdominal pain. Onset was 8-10 hours ago in the right lower quadrant without radiation. Started as 4/10 and is now 5/10, constant,dull, achy, and cramping in nature. Tried Ibuprofen for the pain which helped a little. Walking makes the pain worse. Associated with 4-5 episodes of non-bloody, non-mucoid diarrhea every day for the past 2-3 days. Has had similar episodes in the past that occur randomly, this episode is the worst. No precipitating factors. Denies past medical conditions, allergies, meds, surgeries, travel history, trauma. No family history of diseases. OB/GYN: menarche at 13, regular periods each month, LMP 2 weeks ago, uses 4 pads/day in the beginning and then less as progresses. Last sexual intercourse was 9 months ago. Negative results for STIs/STDs in the past. Denies tobacco and rec drug use. Occasionally dirnks alcohol 2x/month. Healthy diet, exercises regularly. Denies nausea, vomiting, fevers, SOB.
# 81-year-old female with a history of emphysema (not on home O2), who presents with three days of shortness of breath thought by her primary care doctor to be a COPD flare. Two days prior to admission, she was started on a prednisone taper and one day prior to admission she required oxygen at home in order to maintain oxygen saturation greater than 90%. She has also been on levofloxacin and nebulizers, and was not getting better, and presented to the Emergency Room. In the Emergency Room, her oxygen saturation was100% on CPAP. She was not able to be weaned off of this despite nebulizer treatment and Solu-Medrol 125 mg IV x2.Review of systems is negative for the following: Fevers, chills, nausea, vomiting, night sweats, change in weight, gastrointestinal complaints, neurologic changes, rashes, palpitations, orthopnea. Is positive for the following: Chest pressure occasionally with shortness of breath with exertion, some shortness of breath that is positionally related, but is improved with nebulizer treatment.

####################

python megatron_gpt_eval.py gpt_model_file=MegatronBERT.nemo server=True

import json
import requests

port_num = 5555
headers = {"Content-Type": "application/json"}

def request_data(data):
    resp = requests.put('http://localhost:{}/generate'.format(port_num),
                        data=json.dumps(data),
                        headers=headers)
    sentences = resp.json()['sentences']
    return sentences


data = {
    "sentences": ["Tell me an interesting fact about space travel."]*1,
    "tokens_to_generate": 50,
    "temperature": 1.0,
    "add_BOS": True,
    "top_k": 0,
    "top_p": 0.9,
    "greedy": False,
    "all_probs": False,
    "repetition_penalty": 1.2,
    "min_tokens_to_generate": 2,
}

sentences = request_data(data)
print(sentences)

37 yo m with a pmh of sinus [MASK] s/p septal surgery and gerd who presented to the ed from clinic with fevers. he reports that he thought he had a uri for the last 3-4 days. then last night he developed high fevers and fatigue. sunday he began feeling as if he had a cold then monday began feeling achy this progressed on teusday and then teusday night he began having shaking chills and back pain. he went to see his pcp's office today. they noted a temperature of 102 and pleuritic chest pain and sent him to the ed for further work up because he was so ill appearing. he also notes a sore throat and productive cough with clear sputum, chills and rigors last night and pleuritic right chest pain. he had some nausea and dizziness as well. in the ed, his initial vital signs were t 98.9, hr 94, bp 96/54, rr 17, o2sat 100. he had a cxr which suggested rll pneumonia. he was ordered for levofloxacin, ceftriaxone and vancomycin as the ed was concerned over community acquired mrsa pneumonia given question of preceeding viral syndrome. lactate was elevated to 2.6; therefore, he was given 3l ns. however, his bps continued to drift to the low 90s when ivf were stopped. thus he is admitted to the hospital unit 1 for further managment. he denies n/v/d, numbness, tingling, and shortness of breath. he does complain of decreased urine output. review of systems is otherwise negative.


from transformers import pipeline

oracle = pipeline("zero-shot-classification", model="./")
oracle(
    "37 yo m with a pmh of sinus headaches s/p septal surgery and gerd who presented to the ed from clinic with fevers. he reports that he thought he had a uri for the last 3-4 days. then last night he developed high fevers and fatigue. sunday he began feeling as if he had a cold then monday began feeling achy this progressed on teusday and then teusday night he began having shaking chills and back pain. he went to see his pcp's office today. they noted a temperature of 102 and pleuritic chest pain and sent him to the ed for further work up because he was so ill appearing. he also notes a sore throat and productive cough with clear sputum, chills and rigors last night and pleuritic right chest pain. he had some nausea and dizziness as well. in the ed, his initial vital signs were t 98.9, hr 94, bp 96/54, rr 17, o2sat 100. he had a cxr which suggested rll pneumonia. he was ordered for levofloxacin, ceftriaxone and vancomycin as the ed was concerned over community acquired mrsa pneumonia given question of preceeding viral syndrome. lactate was elevated to 2.6; therefore, he was given 3l ns. however, his bps continued to drift to the low 90s when ivf were stopped. thus he is admitted to the hospital unit 1 for further managment. he denies n/v/d, numbness, tingling, and shortness of breath. he does complain of decreased urine output. review of systems is otherwise negative.",
    candidate_labels=["head", "chest", "abdomen", "extremities", "other"],
)

oracle(
    "I have a problem with my iphone that needs to be resolved asap!!",
    candidate_labels=["english", "german"],
)

from transformers import GPTN