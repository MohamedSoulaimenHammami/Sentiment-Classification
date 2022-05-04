# !pip install torch
# !pip install transformers
# !pip install pandas
# !pip install transformers[sentencepiece]
# !pip install emoji
# !pip install datasets
import torch
import numpy as np 
from transformers import AutoTokenizer , AutoModelForSequenceClassification 
import pandas as pd
from torch.utils.data import Dataset 
from sklearn.model_selection import train_test_split
from transformers import Trainer , TrainingArguments
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader
from transformers import  AdamW
import os 
from datasets import load_metric

######################################################################

#Dans ce fichier vous devrez changer le path de NLP 

######################################################################

NLP='/home/msh/Desktop/NLP/nlp/NLP'
Paths={
    'fine_tuned1':os.path.join(NLP , 'fine_tuned1'),
    'fine_tuned2':os.path.join(NLP,'fine_tuned2')
}

for path in Paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            # !mkdir -p {path}
            os.makedirs(path)
        if os.name == 'nt':
            os.makedirs(path)

def read_csv (path) : 
  all_data = pd.read_csv(path)
  return all_data
def spliting_data(all_data):
  negative_data=all_data[:47]
  positive_data=all_data[47:]
  train_neg_v1=negative_data[:20]
  train_neg_v2=negative_data[:40]
  test_negative=negative_data[40:]
  train_pos_v1=positive_data[:200]
  train_pos_v2=positive_data[200:500]
  test_positive=positive_data[500:]
  test_set=pd.concat([test_positive,test_negative]) # test set for all  de longeur =43 
  train_set_v1=pd.concat([train_neg_v1,train_pos_v1]) # train set for fine tunning version 1 longeur 220 (neg: 20 , pos : 200)
  train_set_v2=pd.concat([train_neg_v2,train_pos_v2]) # train set for fine tunning version 2 longeur 340 (neg : 40 , pos : 300)
  return test_set , train_set_v1 , train_set_v2
# all_data = pd.read_csv('/content/gdrive/MyDrive/combined.csv')


# liste des models :  
models=["cardiffnlp/twitter-roberta-base-sentiment","cardiffnlp/twitter-xlm-roberta-base-sentiment",
       "philschmid/distilbert-base-multilingual-cased-sentiment","philschmid/distilbert-base-multilingual-cased-sentiment-2",
       "Monsia/camembert-fr-covid-tweet-sentiment-classification",
       "moussaKam/barthez-sentiment-classification","siebert/sentiment-roberta-large-english"
      
]

# Create Pytorch Dataset
class ourDataset(Dataset):
  def __init__(self,encodings,labels):
    self.encodings=encodings
    self.labels=labels
  def __getitem__(self, idx):
    item={key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels']=torch.tensor(self.labels[idx])
    return item
  def __len__(self):
    return len(self.labels)



def fine_tunning(model,train_data,test_data,Path):
  # model yekhou ism model , data tekhou train set 1 ou  2
  train_texts=train_data['content'].values.tolist()
  train_labels=train_data['sentiment'].values.tolist()
  test_texts=test_data['content'].values.tolist()
  test_labels=test_data['sentiment'].values.tolist()
  # splitting between train and test 
  train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
  model_name=model
  # Defining our tokenizer // remarque fil video ista3mel DistilBertTokenizerFast : car heka mich yesta3mlou 
  # 3ala custom model  so  i used the AutoTokenize pour éviter toute sorte de problème  
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  #Encoding the train , the val and also the test sets 
  train_encodings = tokenizer(train_texts,truncation= True , padding= True , max_length= 512, return_tensors='pt' )
  val_encodings = tokenizer(val_texts , truncation= True , padding= True , max_length= 512 )
  test_encodings = tokenizer(test_texts , truncation= True , padding= True , max_length= 512 )
  # Create our pytorch dataset 
  train_dataset=ourDataset(train_encodings , train_labels)
  val_dataset= ourDataset(val_encodings , val_labels)
  # test_dataset= ourDataset(test_encodings,test_labels )
  model=AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
  torch.cuda.empty_cache()
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model.to(device)
  model.train()
  train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
  optim = AdamW(model.parameters(), lr=5e-5)
  save=os.path.join(Path,model_name)
  for epoch in range(30):
      for batch in train_loader:
          optim.zero_grad()
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['labels'].to(device)
          outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
          loss = outputs[0]
          loss.backward()
          optim.step()
  model.save_pretrained(save)
  tokenizer.save_pretrained(save)
  return test_texts , test_labels
def evaluate_model(model_path,test_texts,test_labels) :
  model=AutoModelForSequenceClassification.from_pretrained(model_path)
  tokenizer=AutoTokenizer.from_pretrained(model_path)
  test_set=tokenizer(test_texts,padding=True , truncation=True ,max_length=512, return_tensors="pt")
  metric=load_metric("accuracy")
  model.eval()
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  
  with torch.no_grad():
      outputs = model(**test_set)
      logits = outputs.logits    
  predictions = torch.argmax(logits, dim=-1)
  metric.add_batch(predictions=predictions,references=test_labels)
  return metric.compute()



def running(models,test_set,train_set_v1,train_set_v2):
    details=dict()
    detail_model=dict()
    for model_name in models :
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)
        test_texts,test_labels=fine_tunning(model_name,train_set_v1,test_set,Paths['fine_tuned1'])
        model_path=os.path.join(Paths['fine_tuned1'],model_name)
        details['accuracy_v1_FT']=evaluate_model(model_path,test_texts,test_labels)
        detail_model[model_name]=details
        test_texts,test_labels=fine_tunning(model_name,train_set_v2,test_set , Paths['fine_tuned2'])
        model_path=os.path.join(Paths['fine_tuned2'],model_name)
        details['accuracy_v2_FT']=evaluate_model(model_path,test_texts,test_labels)
        detail_model[model_name]=details
    return details


######################################################################

# Vous devrez changer le path de file csv

######################################################################

all_data=read_csv('/home/msh/Desktop/NLP/nlp/NLP/combined.csv')
test_set , train_set_v1 , train_set_v2= spliting_data(all_data)
details=running(models , test_set , train_set_v1, train_set_v2)