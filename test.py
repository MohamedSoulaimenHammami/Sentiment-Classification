import torch.nn.functional as F
import torch 
from transformers import AutoTokenizer , AutoModelForSequenceClassification
import os
import warnings
warnings.filterwarnings('ignore')
import time 


######################################################################

# Dans ce fichier vous devrez changer le path de NLP 


######################################################################

models=["cardiffnlp/twitter-roberta-base-sentiment","cardiffnlp/twitter-xlm-roberta-base-sentiment",
       "philschmid/distilbert-base-multilingual-cased-sentiment","philschmid/distilbert-base-multilingual-cased-sentiment-2",
       "Monsia/camembert-fr-covid-tweet-sentiment-classification",
       "moussaKam/barthez-sentiment-classification","siebert/sentiment-roberta-large-english"
      
]

NLP='/home/msh/Desktop/NLP/nlp/NLP'
Paths={
    'fine_tuned1':os.path.join(NLP , 'fine_tuned1'),
    'fine_tuned2':os.path.join(NLP,'fine_tuned2')
}
def testing(model_name , comment):
    model_FT1=os.path.join(Paths['fine_tuned1'], model_name)
    model_FT2=os.path.join(Paths['fine_tuned2'],model_name)
    tokenizer_FT1=AutoTokenizer.from_pretrained(model_FT1)
    tokenizer_FT2=AutoTokenizer.from_pretrained(model_FT2)
    modelFT1=AutoModelForSequenceClassification.from_pretrained(model_FT1)
    modelFT2=AutoModelForSequenceClassification.from_pretrained(model_FT2)
    batch_FT1=tokenizer_FT1(comment,padding=True ,truncation=True , max_length= 512 , return_tensors='pt')
    batch_FT2=tokenizer_FT2(comment,padding=True ,truncation=True , max_length= 512 , return_tensors='pt')
    start=time.time()
    with torch.no_grad() :
        outputs=modelFT1(**batch_FT1)
        prediction=F.softmax(outputs.logits, dim=1)
        labels=torch.argmax(prediction , dim= 1)
        labels=[modelFT1.config.id2label[label_id] for label_id in labels.tolist()]
        end1=time.time()
        outputs2=modelFT2(**batch_FT2)
        prediction2=F.softmax(outputs2.logits, dim=1)
        labels2=torch.argmax(prediction2 , dim= 1)
        labels2=[modelFT2.config.id2label[label_id] for label_id in labels2.tolist()]
    end=time.time()
    temps_exec1=end1-start
    temps_exec2=end -end1
    print("le temps d'éxécution de version du model aprés fine tuning 1 est: {} \nle temps d'éxécution du model aprés fine tuning 2 est: {}".format(temps_exec1, temps_exec2))
    return labels , labels2

comment= input("donner votre commentaire pour le tester :\n")
for model_name in models :
    label_FT1,label_FT2= testing(model_name , comment)
    print("la valeur d'accurarcy pour le comment :\n{} \nest pour le fine tuning 1 {} et pour le fine tuning 2 {}".format(comment , label_FT1 , label_FT2)) 



# def testing(model_name , comment):
#     model_FT1=os.path.join(Paths['fine_tuned1'], model_name)
#     model_FT2=os.path.join(Paths['fine_tuned2'],model_name)
#     tokenizer_FT1=AutoTokenizer.from_pretrained(model_FT1)
#     tokenizer_FT2=AutoTokenizer.from_pretrained(model_FT2)
#     modelFT1=AutoModelForSequenceClassification.from_pretrained(model_FT1)
#     modelFT2=AutoModelForSequenceClassification.from_pretrained(model_FT2)
#     batch_FT1=tokenizer_FT1(comment,padding=True ,truncation=True , max_length= 512 , return_tensors='pt')
#     batch_FT2=tokenizer_FT2(comment,padding=True ,truncation=True , max_length= 512 , return_tensors='pt')
#     start=time.time()
#     from transformers import pipeline
#     classifier1=pipeline("sentiment-analysis",model=modelFT1 ,tokenizer=tokenizer_FT1)
#     return classifier1(comment)


