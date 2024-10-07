import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np
from tqdm import tqdm
from utils import produce_NA, get_main_device
import pickle
import random

class RawData(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

def create_edge_node(df):
    nrow, ncol = df.shape
    # if ncol < 36:
    #     n_target = 36
    # elif ncol < 64:
    #     n_target = 64
    # elif ncol < 128:
    #     n_target = 128
    # elif ncol < 256:
    #     n_target = 256
    # elif ncol < 512:
    #     n_target = 512
    # else:
    #     n_target = 1024
    
    # if ncol < 32:
    #     n_target = 32
    # else:
    #     n_target = 64
    n_target = 32

    if ncol < 32:
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,n_target))
        feature_node[np.arange(ncol), feature_ind+1] = 1
    else:
        feature_node = np.zeros((ncol, n_target))
        for i in range(ncol):
            feature_node[i, i % n_target] = 1
            feature_node[i, (i + n_target // 2) % n_target] = 1
    sample_node = np.zeros((nrow,n_target))
    sample_node[:,0] = 1
    node = sample_node.tolist() + feature_node.tolist()
    return node

def create_value_node(df):
    nrow, ncol = df.shape
    value_node = []
    for i in range(nrow):
        for j in range(ncol):
            # value_node.append([float(df.iloc[i,j])])
            value_node.append([df.iloc[i,j]])
    value_node = value_node + value_node
    return value_node

def create_VE_affiliation(df):
    n_row, n_col = df.shape
    start = []
    end = []
    for x in range(n_row):
        start = start + [x] * n_col  # row-level hyper-edge
        end = end + list(n_row+np.arange(n_col)) # column-level hyper-edge
    start_dup = start + end
    end_dup = end + start
    return torch.tensor([start_dup, end_dup], dtype=int)

def get_data(df_X, missing_ratio, missing_mechanism, seed=0, normalize=True):
    
    hyperedge = create_edge_node(df_X) 
    hyperedge = torch.tensor(hyperedge, dtype=torch.float)

    hyper_node = create_value_node(df_X) 
    hyper_node = torch.tensor(hyper_node, dtype=torch.float)

    ve_affiliation = create_VE_affiliation(df_X)

    torch.manual_seed(seed)

    # train_mask = get_known_mask(1-missing_ratio, int(hyper_node.shape[0]/2))
    # Introduce missing data
    if missing_mechanism == "MCAR":
        train_mask = produce_NA(hyper_node[:int(hyper_node.shape[0]/2)], p_miss=missing_ratio, mecha="MCAR", n_row=df_X.shape[0], n_col=df_X.shape[1])
    elif missing_mechanism == "MAR":
        train_mask = produce_NA(hyper_node[:int(hyper_node.shape[0]/2)], p_miss=missing_ratio, mecha="MAR", n_row=df_X.shape[0], n_col=df_X.shape[1], p_obs=0.5)
    elif missing_mechanism == "MNAR":
        # train_mask = produce_NA(hyper_node[:int(hyper_node.shape[0]/2)], p_miss=missing_ratio, mecha="MNAR", n_row=df_X.shape[0], n_col=df_X.shape[1], opt="logistic", p_obs=0.5, q=0.3)
        train_mask = produce_NA(hyper_node[:int(hyper_node.shape[0]/2)], p_miss=missing_ratio, mecha="MNAR", n_row=df_X.shape[0], n_col=df_X.shape[1], opt="logistic", p_obs=0.5, q=0.3)
    else:
        raise ValueError("Missing mechanism not implemented")
    train_mask_dup = torch.cat((train_mask, train_mask), dim=0)

    train_hyper_node = hyper_node.clone().detach()
    train_ve_affiliation = ve_affiliation.clone().detach()
    train_hyper_node = train_hyper_node[train_mask_dup]
    train_ve_affiliation = train_ve_affiliation[:,train_mask_dup]
    train_labels = train_hyper_node[:int(train_hyper_node.shape[0]/2),0]


    test_hyper_node = hyper_node.clone().detach()
    test_ve_affiliation = ve_affiliation.clone().detach()
    test_hyper_node = test_hyper_node[~train_mask_dup]
    test_ve_affiliation = test_ve_affiliation[:,~train_mask_dup]
    test_labels = test_hyper_node[:int(test_hyper_node.shape[0]/2),0]

    return hyperedge, train_hyper_node, train_ve_affiliation, train_labels, test_hyper_node, test_ve_affiliation, test_labels 

def encode(texts, tokenizer, model, bs_embedding=256):

    result = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    input_ids = result.input_ids
    attention_mask = result.attention_mask

    text_dataset = RawData(input_ids, attention_mask)
    
    dataloader = DataLoader(
        text_dataset,
        batch_size=bs_embedding,
        shuffle=False,
        num_workers=4
    )

    hidden_states_total = []
    sentence_embedding_total = []
    labels_total = []

    device = get_main_device(model)

    for batch in dataloader:
        # print(batch)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Prepare labels (here we use the offset version of the input as labels).
        labels = input_ids.clone()
        labels = torch.roll(labels, shifts=-1, dims=1)
        labels[:, -1] = -100  # The last token has no next word, so we use -100.

        # We also need to set the labels for the padding positions to -100.
        labels = labels.masked_fill(attention_mask == 0, -100)

        hidden_states = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        ).last_hidden_state

        # Use the embedding of the last token as the sentence embedding.
        sentence_embedding = hidden_states[:, -1]  

        hidden_states_total.append(hidden_states.cpu().detach())
        sentence_embedding_total.append(sentence_embedding.cpu().detach())
        labels_total.append(labels.cpu().detach())

    token_emb = torch.cat(hidden_states_total, dim=0)
    sentence_emb = torch.cat(sentence_embedding_total, dim=0)
    labels = torch.cat(labels_total, dim=0)

    return sentence_emb, token_emb, labels

def create_edge_node_llm(df_X, llm_model, tokenizer):
    row_level_info = [f"this is row: {i}" for i in range(df_X.shape[0])]
    col_level_info = [f"this is col: {i}" for i in df_X.columns.tolist()]
    # print(row_level_info, col_level_info)
    row_emb, _, _ = encode(row_level_info, tokenizer, llm_model)
    col_emb, _, _ = encode(col_level_info, tokenizer, llm_model)

    return torch.cat((row_emb, col_emb), dim=0)

def build_cell_info_string(df, row_index, col_name, train_mask, dataset_name):
    row = df.iloc[row_index]
    cell_value = row[col_name]
    
    # other_cols = [f"{col}: {row[col]}" for col in df.columns if col != col_name]
    other_cols = []
    for col in df.columns:
        if col != col_name:
            if train_mask[row_index, df.columns.get_loc(col)]:
                other_cols.append(f"{col} {row[col]}")
            else:
                other_cols.append(f"{col} NaN")
    
    random.shuffle(other_cols)
    other_cols_str = ", ".join(other_cols)
    
    # result = f"information of row {row_index} in dataset of {dataset_name}, {other_cols_str}, {col_name} {cell_value} <eos>"
    # result_prefix = f"information of row {row_index} in dataset of {dataset_name}, {other_cols_str}, {col_name} "
    result = f"row {row_index}, Given {other_cols_str}, Question: {col_name} => {cell_value} <eos>"
    result_prefix = f"row {row_index}, Given {other_cols_str}. Question: {col_name} =>"

    return result, result_prefix


def create_value_node_llm(df, llm_model, tokenizer, dataset_name, train_mask):
    
    nrow, ncol = df.shape
    value_node = []
    value_node_prefix = []

    column_names = df.columns.tolist()
    if dataset_name == "buy":
        prefix = ["Example: [row 2, col name, dataset buy] => Netgear ProSafe FS105 Ethernet Switch - FS105NA. [row 642, col name, dataset buy] => Apple 8x DVDRW Drive - MB397G/A; [row 343, col name, dataset buy] => Panasonic Viera TH-50PZ80U 50' Plasma TV; ",
                "Example: [row 3, col description, dataset buy] => 1 x HD-15 - 1 x HD-15 - 10ft - Beige; [row 343, col description, dataset buy] => 50' - ATSC, NTSC - 16:9 - 1920 x 1080 - Surround - HDTV; [row 265, col description, dataset buy] => TV, Cable Box - Universal Remote; ",
                "Example: [row 3, col manufacturer, dataset buy] => Sony; [row 289, col manufacturer, dataset buy] => Apple; [row 292, col manufacturer, dataset buy] => Yamaha; ",
                "Example: [row 6, col price, dataset buy] => $9.99; [row 298, col price, dataset buy] => $86.95; [row 595, col price, dataset buy] => $11.99;"]
       
    else:
        prefix = ["Example: [row 10, col name, dataset restaurant] => newsbar; [row 642, col name, dataset restaurant] => yoyo tsumami bistro; [row 863, col name, dataset restaurant] => wa-ha-ka oaxaca mexican grill; ",
                "Example: [row 13, col addr, dataset restaurant] => 57 w. 48th st; [row 769, col addr, dataset restaurant] => 3700 w. flamingo rd.; [row 201, col addr, dataset restaurant] => 777 sutter st.; ",
                "Example: [row 4, col city, dataset restaurant] => new york; [row 536, col city, dataset restaurant] => las vegas; [row 281, col city, dataset restaurant] => los angeles; ",
                "Example: [row 2, col phone, dataset restaurant] => 212/679-5535; [row 544, col phone, dataset restaurant] => 702/735-8686; [row 747, col phone, dataset restaurant] => 718-858-4300; ",
                "Example: [row 10, col type, dataset restaurant] => coffee bar; [row 307, col type, dataset restaurant] => italian; [row 863, col type, dataset restaurant] => mexican; "]
        
    for i in range(nrow):
        for j in range(ncol):
            
            cell_info, cell_prefix = build_cell_info_string(df, i, column_names[j], train_mask, dataset_name)
            # print(cell_info)
            # print(cell_prefix)
            value_node.append(cell_info)
            value_node_prefix.append(cell_prefix)

            # value_node.append(prefix[j]+f"Query: [row {i}, col {column_names[j]}, dataset {dataset_name}] => {df.iloc[i,j]};")
            # value_node_prefix.append(prefix[j]+f"Query: [row {i}, col {column_names[j]}, dataset {dataset_name}] =>")
    
    value_node_prefix = value_node_prefix + value_node_prefix
    sentence_emb, token_emb, labels = encode(value_node, tokenizer, llm_model)
    
    sentence_emb = torch.cat([sentence_emb, sentence_emb], dim=0)
    token_emb = torch.cat([token_emb, token_emb], dim=0)
    labels = torch.cat([labels, labels], dim=0)
    
    return sentence_emb, token_emb, labels, value_node_prefix

def get_data_llm(df_X, dataset_name, llm_model, tokenizer, missing_ratio, missing_mechanism, seed=0, normalize=True):
    
    hyperedge = create_edge_node_llm(df_X, llm_model, tokenizer) 

    hyper_node_value = create_value_node(df_X) 
    
    ve_affiliation = create_VE_affiliation(df_X)

    torch.manual_seed(seed)

    if missing_mechanism == "MCAR":
        train_mask = produce_NA(df_X, p_miss=missing_ratio, mecha="MCAR", n_row=df_X.shape[0], n_col=df_X.shape[1])
    else:
        raise ValueError(f"Missing mechanism {missing_mechanism} not implemented for String Type Data")
    train_mask_dup = torch.cat((train_mask, train_mask), dim=0)

    hyper_node, token_emb, labels, value_node_prefix = create_value_node_llm(df_X, llm_model, tokenizer, dataset_name, train_mask.view(df_X.shape))
    hyper_node = torch.tensor(hyper_node, dtype=torch.float)

    train_hyper_node = hyper_node.clone().detach()
    train_ve_affiliation = ve_affiliation.clone().detach()
    train_hyper_node = train_hyper_node[train_mask_dup]
    train_ve_affiliation = train_ve_affiliation[:,train_mask_dup]
    train_labels = labels[:int(train_hyper_node.shape[0]/2)]
    train_tokens_emb = token_emb[:int(train_hyper_node.shape[0]/2)]

    test_hyper_node = hyper_node.clone().detach()
    test_ve_affiliation = ve_affiliation.clone().detach()
    test_hyper_node = test_hyper_node[~train_mask_dup]
    test_ve_affiliation = test_ve_affiliation[:,~train_mask_dup]
    
    # test_node_text = value_node_prefix[~train_mask_dup]
    test_indices = (~train_mask_dup).nonzero().squeeze()
    test_node_text = [value_node_prefix[i] for i in test_indices]
    test_labels = [hyper_node_value[i] for i in test_indices]

    return hyperedge, train_hyper_node, train_ve_affiliation, train_labels, test_hyper_node, test_ve_affiliation, test_labels, train_tokens_emb, test_node_text


def chunk_dataframe(df, chunk_size):
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
    for i in range(num_chunks):
        yield df[i*chunk_size:(i+1)*chunk_size]

def load_list(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_list(mixed_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(mixed_list, f)

def load_data(args):
   
    # if args.header_type == "LLM":
    #     dataset = ["buy", "restaurant"]
    # else:
    #     dataset = ["wine", "heart", "breast", "car", "wireless", "abalone", "turkiye", "letter", "chess", "shuttle", "yeast", "spam", "phishing"]
        # news to be determined
    # dataset = ["wine", "heart"]
    # dataset = ["wine", "heart", "breast"]
    # dataset = ["buy", "restaurant"]
    # dataset = ["buy"]
    # dataset = ["restaurant"]
    # dataset = ["restaurant_test"]
    # dataset = [args.data]
    if args.mode == "training" or args.mode == "testing":
        if args.header_type == "LLM":
            # dataset = ["buy", "restaurant", "walmart"]
            dataset = ["drug_test", "guitar_test", "flipkart_test", "SMS_test"]
            # dataset = ["buy_test", "restaurant_test", "walmart_test"]
        else:
            dataset = ["parkinsons", "libras", "phishing", "bike", "chess", "shuttle", "power_consumption"]
            # dataset  = ["slump", "iris", "wine", "heart", "yacht", "ionosphere", "climate", "credit", "breast", "blood", "raisin", "review", "health", "compression", "yeast", "airfoil", "car", "drug", "wireless", "obesity", "abalone", "spam", "turkiye", "letter", "news", "connect"]
            # dataset = ["metro"]
    else:
        dataset = [args.data]
        
    hyperedge_all = []
    train_hyper_node_all = []
    train_ve_affiliation_all = []
    train_labels_all = []
    test_hyper_node_all = []
    test_ve_affiliation_all = []
    test_labels_all = [] 
    train_tokens_emb_all = []
    test_node_text_all = []

    if args.header_type == "LLM":
        llm_path = args.llm_path
        llm_model = AutoModelForCausalLM.from_pretrained(llm_path, device_map="auto")

        tokenizer = AutoTokenizer.from_pretrained(llm_path, device_map="auto")
        # tokenizer.pad_token='[PAD]' 
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        for param in llm_model.parameters():
            param.requires_grad = False
    else:
        llm_model = None
        tokenizer = None

    chunk_size = args.chunk_size # 32 for LLM, 500 for Linear
    chunk_map = []

    
    save_path = "./prompt_embedding/"
    if args.load_emb:
        hyperedge_all = load_list(save_path+"hyperedge_all.pkl")
        train_hyper_node_all = load_list(save_path+"train_hyper_node_all.pkl")
        train_ve_affiliation_all = load_list(save_path+"train_ve_affiliation_all.pkl")
        train_labels_all = load_list(save_path+"train_labels_all.pkl")
        test_hyper_node_all = load_list(save_path+"test_hyper_node_all.pkl")
        test_ve_affiliation_all = load_list(save_path+"test_ve_affiliation_all.pkl")
        test_labels_all = load_list(save_path+"test_labels_all.pkl")
        chunk_map = load_list(save_path+"chunk_map.pkl")
        train_tokens_emb_all = load_list(save_path+"train_tokens_emb_all.pkl")
        test_node_text_all = load_list(save_path+"test_node_text_all.pkl")

        return llm_model, tokenizer, hyperedge_all, train_hyper_node_all, train_ve_affiliation_all, train_labels_all, test_hyper_node_all, test_ve_affiliation_all, test_labels_all, dataset, chunk_map, train_tokens_emb_all, test_node_text_all

    for i in tqdm(range(len(dataset))):
        # data_path = "./data/" + dataset[i] + ".csv"
        data_path = "../SIGMOD25_Exp/data_update/" + dataset[i] + ".csv"
        df_X = pd.read_csv(data_path, index_col=None)

        # normalize data
        if args.header_type == "Linear":
            x = df_X.values
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df_X = pd.DataFrame(x_scaled)

        for j, chunk in enumerate(chunk_dataframe(df_X, chunk_size)):
            if args.header_type == "Linear":
                hyperedge, train_hyper_node, train_ve_affiliation, train_labels, test_hyper_node, test_ve_affiliation, test_labels = get_data(df_X=chunk, missing_ratio=args.missing_ratio, missing_mechanism=args.missing_mechanism, seed=args.seed)
            elif args.header_type == "LLM":
                hyperedge, train_hyper_node, train_ve_affiliation, train_labels, test_hyper_node, test_ve_affiliation, test_labels, train_tokens_emb, test_node_text = get_data_llm(df_X=chunk, dataset_name=dataset[i], llm_model=llm_model, tokenizer=tokenizer, missing_ratio=args.missing_ratio, missing_mechanism=args.missing_mechanism, seed=args.seed)
                train_tokens_emb_all.append(train_tokens_emb)
                test_node_text_all.append(test_node_text)
            else:
                raise ValueError("Not supported header type")

            hyperedge_all.append(hyperedge)
            train_hyper_node_all.append(train_hyper_node)
            train_ve_affiliation_all.append(train_ve_affiliation)
            train_labels_all.append(train_labels)
            test_hyper_node_all.append(test_hyper_node)
            test_ve_affiliation_all.append(test_ve_affiliation)
            test_labels_all.append(test_labels)

            chunk_map.append(i)
    
    if args.save_emb:
        save_list(hyperedge_all, save_path+"hyperedge_all.pkl")
        save_list(train_hyper_node_all, save_path+"train_hyper_node_all.pkl")
        save_list(train_ve_affiliation_all, save_path+"train_ve_affiliation_all.pkl")
        save_list(train_labels_all, save_path+"train_labels_all.pkl")
        save_list(test_hyper_node_all, save_path+"test_hyper_node_all.pkl")
        save_list(test_ve_affiliation_all, save_path+"test_ve_affiliation_all.pkl")
        save_list(test_labels_all, save_path+"test_labels_all.pkl")
        save_list(chunk_map, save_path+"chunk_map.pkl")
        save_list(train_tokens_emb_all, save_path+"train_tokens_emb_all.pkl")
        save_list(test_node_text_all, save_path+"test_node_text_all.pkl")

    
    return llm_model, tokenizer, hyperedge_all, train_hyper_node_all, train_ve_affiliation_all, train_labels_all, test_hyper_node_all, test_ve_affiliation_all, test_labels_all, dataset, chunk_map, train_tokens_emb_all, test_node_text_all
