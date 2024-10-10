# Some modules are from https://github.com/maxiaoba/GRAPE
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.gen_imp import get_gen_imp
from models.imputaion_model import LinearHead, LLMHead
from utils import produce_NA, get_main_device, compute_LLM_generation_metrics
import time
import matplotlib.pyplot as plt
from data_loader import load_data
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import copy
from tqdm import tqdm

class HyperBatch:
    def __init__(self, train_hyper_node, hyperedge, train_ve_affiliation, train_labels, batch, train_tokens_emb):
        self.train_hyper_node = train_hyper_node
        self.hyperedge = hyperedge
        self.train_ve_affiliation = train_ve_affiliation
        self.train_labels = train_labels
        self.batch = batch
        self.train_tokens_emb = train_tokens_emb

    @staticmethod
    def from_data_list(ids, train_hyper_node_all, hyperedge_all, train_ve_affiliation_all, train_labels_all, train_tokens_emb_all):
        batch_train_hyper_node = []
        batch_hyperedge = []
        batch_train_ve_affiliation = []
        batch_train_labels = []
        batch_indicator = []
        batch_train_tokens_emb = []

        cumulative_edge = 0

        for i in range(len(ids)):

            num_edge = hyperedge_all[ids[i]].size(0)
            num_node = train_hyper_node_all[ids[i]].size(0)
            # hyper_node
            batch_train_hyper_node.append(train_hyper_node_all[ids[i]][:int(num_node/2)])
            # batch_train_hyper_node.append(train_hyper_node_all[ids[i]])

            # train_tokens_emb, LinearHead does not have train_tokens_emb
            if len(train_tokens_emb_all)>0:
                batch_train_tokens_emb.append(train_tokens_emb_all[ids[i]])

            # hyper_node
            batch_hyperedge.append(hyperedge_all[ids[i]])

            train_ve_affiliation = train_ve_affiliation_all[ids[i]][:, :int(num_node/2)] + cumulative_edge
            # train_ve_affiliation = train_ve_affiliation_all[ids[i]]+ cumulative_edge
            
            batch_train_ve_affiliation.append(train_ve_affiliation)

            batch_train_labels.append(train_labels_all[ids[i]])

            batch_indicator.append(torch.full((num_edge,), i, dtype=torch.long))

            cumulative_edge += num_edge


        train_hyper_node = torch.cat(batch_train_hyper_node, dim=0)
        hyperedge = torch.cat(batch_hyperedge, dim=0)
        train_ve_affiliation = torch.cat(batch_train_ve_affiliation, dim=1)
        train_labels = torch.cat(batch_train_labels, dim=0)
        batch = torch.cat(batch_indicator)
        if len(batch_train_tokens_emb) > 0:
            train_tokens_emb = torch.cat(batch_train_tokens_emb, dim=0)
        else:
            train_tokens_emb = []

        # undirected
        train_ve_affiliation_reverse = train_ve_affiliation[[1, 0], :]
        train_ve_affiliation = torch.cat([train_ve_affiliation, train_ve_affiliation_reverse], dim=1)
        train_hyper_node = torch.cat([train_hyper_node, train_hyper_node], dim=0)

        return HyperBatch(train_hyper_node, hyperedge, train_ve_affiliation, train_labels, batch, train_tokens_emb)

    def to(self, device):
        self.train_hyper_node = self.train_hyper_node.to(device)
        self.hyperedge = self.hyperedge.to(device)
        self.train_ve_affiliation = self.train_ve_affiliation.to(device)
        self.train_labels = self.train_labels.to(device)
        self.batch = self.batch.to(device)
        return self


# Generate Imputation Through Auto-Regressive
def generate_impute(args, embedding, impute_model, test_ve_affiliation, lm_model, tokenizer, x_text_test, max_new_tokens=16):

    impute_model.eval()
    lm_model.eval()

    batch_size = len(x_text_test)

    if batch_size == 0:
        return [], []

    inputs = tokenizer(x_text_test, padding=True, truncation=True, return_tensors="pt")
    device = get_main_device(lm_model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    new_text = []

    with torch.no_grad():
        generated = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        for _ in range(max_new_tokens):
            generated = generated.to(device)
            attention_mask = attention_mask.to(device)

            outputs = lm_model.model(
                input_ids=generated,
                attention_mask=attention_mask,
                return_dict=True
            )

            hidden_states = outputs.last_hidden_state.to(f'cuda:{args.device}')
            
            logits = impute_model([embedding[test_ve_affiliation[0]], embedding[test_ve_affiliation[1]]], hidden_states)

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            generated = torch.cat([generated, next_token.unsqueeze(-1).to(device)], dim=-1)
            new_text.append(next_token.unsqueeze(-1).to(device))
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device)], dim=-1)
            
            if all(next_token == tokenizer.eos_token_id):
                break
    
    # Concatenate new_text tensors along the sequence dimension
    new_text_tensor = torch.cat(new_text, dim=1)
    
    return [tokenizer.decode(gen.squeeze(), skip_special_tokens=True) for gen in new_text_tensor], [tokenizer.decode(gen, skip_special_tokens=True) for gen in generated]

def plot(data, figure_name):

    plt.figure(figsize=(10, 6))
    plt.plot(data)

    min_value = min(data)
    last_value = data[-1]
    min_index = data.index(min_value)
    last_index = len(data) - 1

    plt.annotate(f'Min: {min_value}', 
                 xy=(min_index, min_value), 
                 xytext=(0.05, 0.95), 
                 textcoords='axes fraction',
                 arrowprops=dict(facecolor='blue', shrink=0.05))

    plt.annotate(f'Last: {last_value}', 
                 xy=(last_index, last_value), 
                 xytext=(0.95, 0.95), 
                 textcoords='axes fraction',
                 ha='right',
                 arrowprops=dict(facecolor='red', shrink=0.05))
    
    plt.title(figure_name)
    plt.xlabel('Index')
    plt.ylabel('Value')

    plt.savefig("./figures/"+figure_name+".png")
    plt.close()


def test_model(args, device=torch.device('cpu')):
    
    lm_model, tokenizer, hyperedge, train_hyper_node, train_ve_affiliation, train_labels, test_hyper_node, test_ve_affiliation, test_labels, dataset, chunk_map, train_tokens_emb, test_node_text = load_data(args)
    
    model = get_gen_imp(hyperedge[0].shape[1], train_hyper_node[0].shape[1], args).to(device)
    
    
    if args.header_type == "Linear":
        # impute_hiddens = list(map(int,args.impute_hiddens.split('_')))
        impute_hiddens = hyperedge[0].shape[1]
        input_dim = args.hyperedge_dim_hidden * 2
        output_dim = 1
        impute_model = LinearHead(input_dim, output_dim,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout).to(device)
        test_labels_all = [item.clone().detach() for item in test_labels]

        # Load saved parameters
        model.load_state_dict(torch.load(f"./saved_models/llm_gnn_model_{args.load_model_name}.pth"))
        impute_model.load_state_dict(torch.load(f"./saved_models/llm_impute_model_{args.load_model_name}.pth"))

    elif args.header_type == "LLM":
        impute_hiddens = hyperedge[0].shape[1]
        input_dim = args.hyperedge_dim_hidden * 2
        output_dim = args.vocab_size # vocab_size for LlamaLite
        impute_model = LLMHead(input_dim, output_dim,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout,
                            relation_type=args.relation_type)
        
        impute_model.lm_head.weight.data = lm_model.lm_head.weight.data.clone()
        if lm_model.lm_head.bias is not None:
            impute_model.lm_head.bias.data = lm_model.lm_head.bias.data.clone()
        else:
            impute_model.lm_head.bias.data = torch.zeros_like(impute_model.lm_head.bias.data)
        impute_model = impute_model.to(device)
        test_labels_all = [copy.deepcopy(item) for item in test_labels]

        # Load saved parameters
        model.load_state_dict(torch.load(f"./saved_models/llm_gnn_model_{args.load_model_name}.pth"))
        impute_model.load_state_dict(torch.load(f"./saved_models/llm_impute_model_{args.load_model_name}.pth"))
    
    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters())
    
    # print(model)
    # print(impute_model)
    # print('total params in GNN model:', sum(p.numel() for p in model.parameters()))
    # print('total params in impute model:', sum(p.numel() for p in impute_model.parameters()))
    

    filter_fn = filter(lambda p : p.requires_grad, trainable_parameters)
    optimizer = torch.optim.AdamW(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    
    train_hyper_node_all = [item.clone().detach() for item in train_hyper_node]
    hyperedge_all = [item.clone().detach() for item in hyperedge]
    train_ve_affiliation_all = [item.clone().detach() for item in train_ve_affiliation]
    train_labels_all = [item.clone().detach() for item in train_labels]
    test_hyper_node_all = [item.clone().detach() for item in test_hyper_node]
    test_ve_affiliation_all = [item.clone().detach() for item in test_ve_affiliation]
    train_tokens_emb_all = [item.clone().detach() for item in train_tokens_emb]

    start_time = time.time()
    loss_all = [[] for i in range(len(dataset))]
    rmse_all = [[] for i in range(len(dataset))]
    mae_all = [[] for i in range(len(dataset))]


    # print(chunk_map)
    train_ids = [i for i in range(len(chunk_map))]
    train_loader = DataLoader(train_ids, batch_size=32, shuffle=False)

    # gnn_state_dict = torch.load("./saved_models/all_gnn_model_weights_batch.pth", map_location=device)
    # impute_state_dict = torch.load("./saved_models/all_impute_model_weights_batch.pth", map_location=device)

    # model.load_state_dict(gnn_state_dict)
    # impute_model.load_state_dict(impute_state_dict)

    model.eval()
    impute_model.eval()


    
    with torch.no_grad():
        for k in range(len(dataset)):
            dataset_chunk = [item==k for item in chunk_map]
            pred_test_all = []
            label_test_all = []
            for i in tqdm(range(len(chunk_map))):
                if not dataset_chunk[i]:
                    continue
                train_hyper_node = train_hyper_node_all[i].to(device)
                hyperedge = hyperedge_all[i].to(device)
                train_ve_affiliation = train_ve_affiliation_all[i].to(device)
                
                test_hyper_node = test_hyper_node_all[i].to(device)
                test_ve_affiliation = test_ve_affiliation_all[i].to(device)
                test_labels = test_labels_all[i]

                embedding, hyper_node = model(hyperedge, train_hyper_node, train_ve_affiliation)
                if args.header_type == "Linear":
                    pred = impute_model([embedding[test_ve_affiliation[0], :], embedding[test_ve_affiliation[1], :]], token_emb=[])
                    pred_test_all.append(pred[:int(test_hyper_node.shape[0] / 2),0])
                    label_test_all.append(test_labels.to(device))
                
                elif args.header_type == "LLM":
                    x_text_test = test_node_text[i]
                    # print(x_text_test)
                    half_length = len(x_text_test) // 2  
                    x_text_test = x_text_test[:half_length]
                    test_ve_affiliation = test_ve_affiliation[:, :half_length]

                    results, full_results = generate_impute(args, embedding, impute_model, test_ve_affiliation, lm_model, 
                                    tokenizer, x_text_test, max_new_tokens=10)
                    pred_test_all.append(results)
                    label_test_all.append(test_labels[:half_length])
                    # for n in range(len(results)):
                    #     print(f"x_text_test[n]: {x_text_test[n]}")
                    #     print(f"results[n]: {results[n]}")
                    #     print(f"test_labels[n]: {test_labels[n]}")
            if args.header_type == "Linear":
                pred_test = torch.cat(pred_test_all)
                label_test = torch.cat(label_test_all)
                mse = F.mse_loss(pred_test, label_test)
                test_rmse = np.sqrt(mse.item())
                l1 = F.l1_loss(pred_test, label_test)
                test_l1 = l1.item()
                print(f"=== {dataset[k]}, the pred_test size is : {pred_test.shape}, the label_test size is : {label_test.shape} ===")  
                print('test rmse: ', test_rmse)
                print('test l1: ', test_l1)
                print(f"training time is : {time.time()-start_time:.4g}s")
                rmse_all[k].append(test_rmse)
                mae_all[k].append(test_l1)
            elif args.header_type == "LLM":
                print(f"=== In the LLM, dataset: {dataset[k]} ===")
                print(f"testing time is : {time.time()-start_time:.4g}s")
                # print(pred_test_all[0][0:5], label_test_all[0][0:5])
                avg_bleu, avg_rouge_1, avg_rouge_l, avg_rouge_lsum, avg_rouge_w, avg_rouge_s, avg_jaccard, avg_levenshtein, avg_cosine, avg_cosine_tf, avg_cosine_tfidf, avg_cosine_word_embeddings = compute_LLM_generation_metrics(pred_test_all, label_test_all)

                print(f"Average BLEU Score: {avg_bleu:.4f}")
                print(f"Average ROUGE-1 Score: {avg_rouge_1:.4f}")
                print(f"Average ROUGE-L Score: {avg_rouge_l:.4f}")
                print(f"Average ROUGE-Lsum Score: {avg_rouge_lsum:.4f}")
                print(f"Average ROUGE-W Score: {avg_rouge_w:.4f}")
                print(f"Average ROUGE-S Score: {avg_rouge_s:.4f}")
                print(f"Average Jaccard Similarity: {avg_jaccard:.4f}")
                print(f"Average Levenshtein Distance: {avg_levenshtein:.4f}")
                print(f"Average Cosine Similarity: {avg_cosine:.4f}")
                print(f"Average Cosine Similarity (TF): {avg_cosine_tf:.4f}")
                print(f"Average Cosine Similarity (TF-IDF): {avg_cosine_tfidf:.4f}")
                print(f"Average Cosine Similarity (Word Embeddings): {avg_cosine_word_embeddings:.4f}")

    
    # for k in range(len(dataset)):
    #     plot(loss_all[k], dataset[k]+"_"+args.plot_name+"_loss_all")
    #     plot(rmse_all[k], dataset[k]+"_"+args.plot_name+"_rmse_all")
    #     plot(mae_all[k], dataset[k]+"_"+args.plot_name+"_mae_all")
    