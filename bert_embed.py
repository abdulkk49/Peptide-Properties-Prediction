import torch, numpy as np
from transformers import BertModel, BertTokenizer
import re
import os
import requests, h5py
from tqdm.auto import tqdm
from os.path import join, exists, dirname, abspath, realpath

def download_file(url, filename):
    response = requests.get(url, stream=True)
    with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                        total=int(response.headers.get('content-length', 0)),
                        desc=filename) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)

if __name__ == "__main__":
    pwd = dirname(realpath("__file__"))
    print("Present Working Directory: ", pwd)
    #Pretrained Model files
    modelUrl = 'https://www.dropbox.com/s/dm3m1o0tsv9terq/pytorch_model.bin?dl=1'
    configUrl = 'https://www.dropbox.com/s/d3yw7v4tvi5f4sk/bert_config.json?dl=1'
    vocabUrl = 'https://www.dropbox.com/s/jvrleji50ql5m5i/vocab.txt?dl=1'

    #Setting folder paths
    downloadFolderPath = 'models/ProtBert/'
    modelFolderPath = downloadFolderPath

    #Setting file paths
    modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')
    configFilePath = os.path.join(modelFolderPath, 'config.json')
    vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')

    #Creading model directory
    if not os.path.exists(modelFolderPath):
        os.makedirs(modelFolderPath)

    #Downloading pretrained model
    if not os.path.exists(modelFilePath):
        download_file(modelUrl, modelFilePath)
    if not os.path.exists(configFilePath):
        download_file(configUrl, configFilePath)
    if not os.path.exists(vocabFilePath):
        download_file(vocabUrl, vocabFilePath)

    #Initializing Tokenizer, Model
    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False )
    model = BertModel.from_pretrained(modelFolderPath)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()

    #Load sequences
    sequences_Example =[]
    count = 0
    with open("./residuesequences.txt", "r") as f:
        for seq in f.readlines():
            desc = str(seq).rstrip('\n')
            sequences_Example.append(desc)
            count += 1
    print("Total data points(Clean): ", str(count))

    #Replace "UZOB" with "X"
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

    #Tokenizing input sequences
    ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, pad_to_max_length=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    #Generating Embeddings
    prefix = join(pwd,"Embeddings")
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    bs = 16
    batch = 0
    count = 0
    i = 0
    embedding = np.zeros((128,1632,1024), dtype = np.float32)

    limit = len(sequences_Example)//bs * bs
    print("Last data point: ", str(limit))
    x = 0
    with torch.no_grad():
        while i + bs <= limit:
            e = model(input_ids=input_ids[i:i+bs],attention_mask=attention_mask[i:i+bs])[0]
            # print(e.shape)
            e = e.cpu()
            embedding[x:x+bs,:,:] = e[:,1:1633,:]
            x += bs
            i += bs
            count += 1
            print(e.shape, count)
            if count % 8 == 0:
                batch += 1
                x = 0
                print("Saving batch " + str(batch))
                embedding_file = join(prefix, "batch" + str(batch) + ".h5")
                print("Batch " + str(batch))
                print(embedding.shape)
                # torch.save(embedding, embedding_file)
                with h5py.File(embedding_file, 'w') as f:
                    f.create_dataset('embed', data=embedding)
                print("Saved...\n")
                embedding.fill(0)

        if i != len(sequences_Example):
            #Final batch < 16
            e = model(input_ids=input_ids[i:len(sequences_Example)],attention_mask=attention_mask[i:len(sequences_Example)])[0]
            e = e.cpu().numpy()
            embedding[x:x+len(sequences_Example) - i,:,:] = e[:,1:1633,:]
            
            batch += 1
            print("Saving batch " + str(batch))
            embedding_file = prefix + str(batch) + ".h5"
            print("Batch " + str(batch))
            print(e.shape)
            # torch.save(embedding, embedding_file)
            with h5py.File(embedding_file, 'w') as f:
                f.create_dataset('embed', data=embedding)
            print("Saved...\n")

    embedding_file = join(prefix, "batch" + str(batch) + ".h5")
    f = h5py.File(embedding_file, 'r')
    arr = f['embed'][()]
    f.close()
    print(arr.shape)
    arr = arr[~(arr == 0).all(1)] # Remove all zero entries
    arr = np.reshape(arr, (-1, 1632, 1024))
    print(arr.shape)
    with h5py.File(join(prefix, "batch85" + ".h5"), 'w') as f:
        f.create_dataset('embed', data=arr)
    f.close()

