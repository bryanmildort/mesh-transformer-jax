import nlpcloud
import pandas as pd
import time

notes = pd.read_csv('notes.csv')
notes = notes.sample(frac = 1)

def nlp_scrape(index):
    global start_index, prompt
    for i in range(index, len(notes)):
        start_index = i
        with open("prompts.txt", 'r') as f:
            prompt = f.read()
        last_ten = f"[Notes]: {'[Notes]: '.join(prompt.split('[Notes]: ')[-9:])}"
        prompt_input = f"{last_ten}\n[Notes]: {notes.iloc[i,0]}\n[Diagnosis]:"
        client = nlpcloud.Client("finetuned-gpt-neox-20b", "1b3fdd3f6f0bbdb70675d4402ca55767d41588ed", gpu=True, lang="en")
        response = client.generation(
            prompt_input,
            min_length=2,
            max_length=15,
            length_no_input=True,
            remove_input=True,
            end_sequence=None,
            top_p=1,
            temperature=0.5,
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
        output = f"\n[Notes]: {notes.iloc[i,0]}\n[Diagnosis]: {response['generated_text']}"
        with open('prompts.txt', 'a') as f:
            f.write(f"{output}")
        time.sleep(30)


start_index = 33999
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

# Convert dataset to zstd compressed json
import json

with open("prompts.txt", 'r') as f:
    prompt = f.read()

prompt_list = prompt.split('[Notes]: ')
prompt_list.pop(0)
output = {}
for i in range(len(prompt_list)):
    row = prompt_list[i].split('[Diagnosis]: ') 
    output['prompt'] = row[0]
    output['completion'] = row[1]
    if (len(output['prompt']) + len(output['completion'])) < 2048:
        with open('dataset.jsonl', 'a') as f:
            f.write('\n')
            json.dump(output, f)




from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset
dataset = load_dataset("text", data_files="mimic_notes.txt")

# with open('dataset.jsonl', 'r') as f:
#     json_list = list(f)

# dataset = load_dataset("json", data_files="dataset.jsonl")

### GPT-J6B ###
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")


### GPT-Neo20B ###
# model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
# tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

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

def tokenize_function(examples):
    return tokenizer(examples["text"], return_tensors='pt', padding=True, truncation=True, max_length=2048)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model.resize_token_embeddings(len(tokenizer)) ##### THIS IS HOW YOU RESIZE THE MODEL TO MATCH THE TOKENIZER!!!!

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(output_dir="test_trainer")
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator
    #eval_dataset=encoded_eval,
    #compute_metrics=compute_metrics,
)

### TO FREE UP SPACE USING LARGE LM ###
import os, shutil
shutil.rmtree("/home/bryan/.cache") 
del dataset
trainer.train()



############


# encoded_input = tokenizer(dataset, padding=True, return_tensors="pt")

# metric = evaluate.load("accuracy")
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# gsutil -m cp -R D:/step/step_383500 gs://gpt-j6b 

from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModel

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
    #do_sample=True,
    temperature=0.5,
    min_length=711,
    max_length=720,
)

gen_text = tokenizer.batch_decode(gen_tokens)[0]

trainer.save_model('./my_model')
tokenizer.save_vocabulary('./my_model')

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
generator = pipeline('text-generation', model='./test_trainer/checkpoint-500')
set_seed(42)
generator(prompt, max_length=50, num_return_sequences=1)

notes_list = list(notes.TEXT)
for i in notes_list:
    with open('mimic_notes.txt', 'a') as f:
        f.write('\n\n##########\n\n')
        f.write(i)   