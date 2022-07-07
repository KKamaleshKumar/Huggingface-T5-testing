import wandb
#wandb.login()
#wandb.init(project="T5-base testing with augmented data", entity="kkamalesh")

from transformers import T5Tokenizer,T5ForConditionalGeneration,DataCollatorForSeq2Seq
from torch.utils.data import Dataset,DataLoader
from torch.nn.functional import relu
from evaluate import load
from datasets import load_metric
import pandas as pd
from collections import Counter
import numpy as np
import torch
from pprint import pprint
from torch import optim
from tqdm import tqdm
#torch.multiprocessing.set_start_method('spawn')
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

BATCH_SIZE=128
NUM_EPOCHS=15

#df_plain=pd.read_csv('sample_text_2000samples_without_attributes.csv')
#df_attributes=pd.read_csv('sample_text_with_attributes.csv')
#df_attributes.rename(columns={'Augmented text':'Augmented Text'},inplace=True)
#df_combined=pd.concat([df_plain[['Augmented Text','Target Text']],df_attributes[['Augmented Text','Target Text']]])
#df_combined=df_combined.sample(frac=1)
#df=df_combined.applymap(lambda x:x.replace('\n',' [LINE] '))

#df_train=df.iloc[:3500,:]
#df_test=df.iloc[3500:,:]
#len_train=df_train.shape[0]
#len_test=df_test.shape[0]

class CreateDataset(Dataset):
    def __init__(self,df):
        source=df.iloc[:,0].values
        target=df.iloc[:,1].values
        self.source_token=tokenizer(list(source),truncation=True)
        self.target_token=tokenizer(list(target),truncation=True)
        self.len=len(source)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,idx):
        #self.source_token.input_ids[idx]=torch.tensor(self.source_token.input_ids[idx],device=device,dtype=torch.float32)
        #self.source_token.attention_mask[idx]=torch.tensor(self.source_token.attention_mask[idx],device=device,dtype=torch.float32)
        #self.target_token.input_ids[idx]=torch.tensor(self.target_token.input_ids[idx],device=device,dtype=torch.float32)

        return {'input_ids':self.source_token.input_ids[idx],
               'attention_mask':self.source_token.attention_mask[idx],'labels': self.target_token.input_ids[idx]}
    
def init_process(rank,world_size, backend='nccl'):
    #print(rank)
    #if rank==0:
     #   os.environ['MASTER_ADDR'] = '12333.0.0.1'
     #   os.environ['MASTER_PORT'] = '8888'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def create_dataloader(rank,world_size,data):
    
    dataset=CreateDataset(data)
    sampler=DistributedSampler(dataset,rank=rank,num_replicas=world_size,shuffle=True)
    dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,sampler=sampler,collate_fn=data_collator)
    
    return dataloader

def cleanup():
    
    dist.destroy_process_group()
    
def reset_metric_train():
    
    global metrics_train
    metrics_train=Counter({'bleu_train':0,'precision_train':0,'recall_train':0,'f-score_train':0})

def reset_metric_test():
    
    global metrics_test
    metrics_test=Counter({'bleu_test':0,'precision_test':0,'recall_test':0,'f-score_test':0})
    
bleu=load('bleu')
rouge=load('rouge')
train_lis=[]
test_lis=[]
def compute_metric(predicted,target,flag,epoch):
    predicted=[tokenizer.decode(x,skip_special_tokens=True) for x in predicted]
    target_bleu=[[tokenizer.decode(relu(x),skip_special_tokens=True)] for x in target]
    target_rouge=[tokenizer.decode(relu(x),skip_special_tokens=True) for x in target]
    bleu_score=bleu.compute(predictions=predicted,references=target_bleu)['bleu']
    rouge_score=rouge.compute(predictions=predicted,references=target_rouge,use_aggregator=False)['rouge1']
    precision=sum([x[0] for x in rouge_score])/BATCH_SIZE
    recall=sum([x[1] for x in rouge_score])/BATCH_SIZE
    fscore=sum([x[2] for x in rouge_score])/BATCH_SIZE
    if flag==0:
        if epoch==NUM_EPOCHS-1:
            train_lis.extend(predicted)               
        return Counter({'bleu_train':bleu_score,'precision_train':precision,'recall_train':recall,'f-score_train':fscore})
    else:        
        if epoch==NUM_EPOCHS-1: 
            test_lis.extend(predicted)
        return Counter({'bleu_test':bleu_score,'precision_test':precision,'recall_test':recall,'f-score_test':fscore})
    
def train(rank,world_size,epoch):
    
    init_process(rank,world_size)
    print(f'GPU {rank+1} training process initialised succesfully\n')
    
    model=T5ForConditionalGeneration.from_pretrained('t5-base')
    model.cuda(rank)
    
    model=DistributedDataParallel(model,device_ids=[rank])
    train_loader=create_dataloader(rank,world_size,df_train)
    test_loader=create_dataloader(rank,world_size,df_test)
    if rank==0:
        pbar=tqdm(range(NUM_EPOCHS))
    dist.barrier()
    optimiser=optim.Adam(model.parameter(),lr=1e-3*world_size)       
    
    for epoch in range(NUM_EPOCHS):
        if rank==0:
            reset_metric_train()
            reset_metric_test()
        dist.barrier()
        model.train()  
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            batch={key:value.cuda(rank) for key,value in batch.items()}
            outputs=model(**batch)
            loss=outputs.loss
            predicted_tokens=model.generate(batch['input_ids'])
            metrics=compute_metric(predicted_tokens,batch['labels'],flag=0,epoch=epoch)
            metrics_train+=metrics
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch={key:value.cuda(rank) for key,value in batch.items()}
                predicted_tokens=model.generate(batch['input_ids'])
                metrics=compute_metric(predicted_tokens,batch['labels'],flag=1,epoch=epoch)
                metrics_test+=metrics
        if rank==0:
            metrics_train={key:values*(BATCH_SIZE/len_train) for key,values in dict(metrics_train).items()}
            metrics_test={key:values*(BATCH_SIZE/len_test) for key,values in dict(metrics_test).items()}
            print('f-score_train: ',metrics_train['f-score_train'],' f-score_test: ',metrics_test['f-score_test'],' bleu_test: ',metrics_test['bleu_test'],' bleu_train: ',metrics_train['bleu_train'])
            #wandb.log ({**metrics_train,**metrics_test})
            global metrics_combined
            metrics_combined={**metrics_train,**metrics_test}
            pbar.update(1)
            
        dist.barrier()
    cleanup()
WORLD_SIZE=torch.cuda.device_count() 
def main():
    mp.spawn(train,args=(WORLD_SIZE,NUM_EPOCHS),nprocs=WORLD_SIZE)
    
    
if __name__=="__main__":
    main()

    
