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


def construct_imputed_chunk(train_ve_affiliation, training_node, test_ve_affiliation, testing_node):
    
    row_num = max(train_ve_affiliation[0, :int(training_node.shape[0]/2)])+1
    col_num = max(train_ve_affiliation[0, int(training_node.shape[0]/2):])-row_num+1
    
    # print(train_ve_affiliation, row_num, col_num)
    imputed_chunk = np.zeros((row_num, col_num))

    for i in range(int(training_node.shape[0]/2)):
        # print(train_ve_affiliation[0, i], train_ve_affiliation[1, i]-row_num, training_node[i])
        imputed_chunk[train_ve_affiliation[0, i], train_ve_affiliation[1, i]-row_num] = training_node[i]

    for i in range(int(testing_node.shape[0]/2)):
        imputed_chunk[test_ve_affiliation[0, i], test_ve_affiliation[1, i]-row_num] = testing_node[i]

    return imputed_chunk

def finetune_model(args, device=torch.device('cpu')):
    
    lm_model, tokenizer, hyperedge, train_hyper_node, train_ve_affiliation, train_labels, test_hyper_node, test_ve_affiliation, test_labels, dataset, chunk_map, train_tokens_emb_LLM, test_node_text = load_data(args)
    
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
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params_impute = [p for p in impute_model.parameters() if p.requires_grad]

    print('total trainable params in GNN model:', sum(p.numel() for p in trainable_params))
    print('total trainable params in impute model:', sum(p.numel() for p in trainable_params_impute))
        

    filter_fn = filter(lambda p : p.requires_grad, trainable_parameters)
    # optimizer = torch.optim.AdamW(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    
    train_hyper_node_all = [item.clone().detach() for item in train_hyper_node]
    hyperedge_all = [item.clone().detach() for item in hyperedge]
    train_ve_affiliation_all = [item.clone().detach() for item in train_ve_affiliation]
    train_labels_all = [item.clone().detach() for item in train_labels]
    test_hyper_node_all = [item.clone().detach() for item in test_hyper_node]
    test_ve_affiliation_all = [item.clone().detach() for item in test_ve_affiliation]
    train_tokens_emb_all = [item.clone().detach() for item in train_tokens_emb_LLM]

    start_time = time.time()
    loss_all = [[] for i in range(len(dataset))]
    rmse_all = [[] for i in range(len(dataset))]
    mae_all = [[] for i in range(len(dataset))]

    # print(chunk_map)
    train_ids = [i for i in range(len(chunk_map))]
    train_loader = DataLoader(train_ids, batch_size=args.chunk_batch, shuffle=False)

    p_miss_ratio = np.linspace(0.65, 0.35, args.epochs)
    if args.data == "power_consumption":
        args.epochs = 100
    
    for epoch in tqdm(range(args.epochs)):
    # for epoch in range(args.epochs):
    
        for ids in train_loader:
            batch = HyperBatch.from_data_list(ids, 
                        train_hyper_node_all, hyperedge_all, 
                        train_ve_affiliation_all, train_labels_all, train_tokens_emb_all)
            train_hyper_node = batch.train_hyper_node.to(device)
            hyperedge = batch.hyperedge.to(device)
            train_ve_affiliation = batch.train_ve_affiliation.to(device)
            train_labels = batch.train_labels.to(device)
            
            model.train()
            impute_model.train()
            optimizer.zero_grad()
            

            known_mask = produce_NA(train_hyper_node[:int(train_hyper_node.shape[0]/2)], p_miss=p_miss_ratio[epoch], mecha="Random")
            # known_mask = produce_NA(train_hyper_node[:int(train_hyper_node.shape[0]/2)], p_miss=1-args.known, mecha="Random")
            known_mask_dup = torch.cat((known_mask, known_mask), dim=0)
            known_hyper_node = train_hyper_node.clone().detach()
            known_ve_affiliation = train_ve_affiliation.clone().detach()
            known_hyper_node = known_hyper_node[known_mask_dup]
            known_ve_affiliation = known_ve_affiliation[:,known_mask_dup]

            embedding, hyper_node = model(hyperedge, known_hyper_node, known_ve_affiliation)
            

            if args.header_type == "Linear":
                train_tokens_emb = batch.train_tokens_emb
                pred = impute_model([embedding[train_ve_affiliation[0, :int(train_hyper_node.shape[0]/2)]], embedding[train_ve_affiliation[1, :int(train_hyper_node.shape[0]/2)]]], train_tokens_emb)
                pred_train = pred[:int(train_hyper_node.shape[0] / 2),0]
                label_train = train_labels
                # huber_loss = torch.nn.HuberLoss(delta=1)  
                huber_loss = torch.nn.HuberLoss(delta=args.delta)
                loss = huber_loss(pred_train, label_train)
            elif args.header_type == "LLM":
                train_tokens_emb = batch.train_tokens_emb.to(device)
                # print(f"the shape of train_tokens_emb is : {train_tokens_emb.shape}, the shape of embedding is : {embedding.shape}, the shape of train_ve_affiliation is : {train_ve_affiliation.shape}, the shape of train_labels is : {train_labels.shape}")
                pred = impute_model([embedding[train_ve_affiliation[0, :int(train_hyper_node.shape[0]/2)]], embedding[train_ve_affiliation[1, :int(train_hyper_node.shape[0]/2)]]], train_tokens_emb)
                pred_train = pred[:int(train_hyper_node.shape[0] / 2)]
                label_train = train_labels
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pred_train.view(-1, pred_train.size(-1)), label_train.view(-1))
            
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

        
        if (epoch+1) % args.eval_epoch_gap ==0: 
            model.eval()
            impute_model.eval()
            with torch.no_grad():
                for k in range(len(dataset)):
                    dataset_chunk = [item==k for item in chunk_map]
                    pred_test_all = []
                    label_test_all = []

                    imputed_chunks = []
                    for i in range(len(chunk_map)):
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

                            # imputed_chunk = construct_imputed_chunk(train_ve_affiliation, train_hyper_node, test_ve_affiliation, pred[:int(test_hyper_node.shape[0] / 2),0])

                            # imputed_chunks.append(imputed_chunk)
                        
                        elif args.header_type == "LLM":
                            x_text_test = test_node_text[i]
                            half_length = len(x_text_test) // 2  
                            x_text_test = x_text_test[:half_length]
                            test_ve_affiliation = test_ve_affiliation[:, :half_length]

                            new_results, full_results = generate_impute(args, embedding, impute_model, test_ve_affiliation,  
                                                      lm_model, tokenizer, x_text_test, max_new_tokens=15)
                            pred_test_all.append(new_results)
                            label_test_all.append(test_labels[:half_length])
                    if args.header_type == "Linear":
                        pred_test = torch.cat(pred_test_all)
                        label_test = torch.cat(label_test_all)
                        mse = F.mse_loss(pred_test, label_test)
                        test_rmse = np.sqrt(mse.item())
                        l1 = F.l1_loss(pred_test, label_test)
                        test_l1 = l1.item()
                        print(f"=== {dataset[k]}, the pred_test size is : {pred_test.shape}, the label_test size is : {label_test.shape} ===")         
                        print('epoch: ', epoch)
                        print('loss: ', train_loss)
                        print('test rmse: ', test_rmse)
                        print('test l1: ', test_l1)
                        print(f"training time is : {time.time()-start_time:.4g}s")
                        loss_all[k].append(train_loss)
                        rmse_all[k].append(test_rmse)
                        mae_all[k].append(test_l1)

                        # imputed_datasets = np.concatenate(imputed_chunks)
                        # np.savetxt(f"../SIGMOD25_Exp/downstream_classification/UnIMP-ft/{dataset[k]}_filled.csv", imputed_datasets, delimiter=",")

                    elif args.header_type == "LLM":
                        print(f"=== In the LLM, epoch: {epoch}, dataset: {dataset[k]} ===")
                        print(f"training time is : {time.time()-start_time:.4g}s")
                        print('loss: ', train_loss)
                        # print(pred_test_all[0][0:5], label_test_all[0][0:5])
                        for m in range(len(pred_test_all)):
                            for n in range(len(pred_test_all[m])):
                                print(f"the query is : {test_node_text[m][n]}, the pred is : {pred_test_all[m][n]}, the label is : {label_test_all[m][n]}")
                        
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

                        
            # torch.save(model.state_dict(), f"./saved_models/llm_gnn_model_{args.header_type}_Epoch{epoch}_{args.save_name}.pth")
            # torch.save(impute_model.state_dict(), f"./saved_models/llm_impute_model_{args.header_type}_Epoch{epoch}_{args.save_name}.pth")
        
    if args.header_type == "Linear":
        print(f"==== Dataset: {dataset} =====")
        print(f"Training Time taken: : {time.time()-start_time:.4g}s")
        print(f"Missing Mechanism: {args.missing_mechanism}, miss_rate: {args.missing_ratio}, RMSE: {rmse_all}, MAE: {mae_all}")
        print(f"Missing Mechanism: {args.missing_mechanism}, miss_rate: {args.missing_ratio}, RMSE: {rmse_all[-1][-1]}, MAE: {mae_all[-1][-1]}")
        # for k in range(len(dataset)):
        #     plot(loss_all[k], dataset[k]+"_"+args.plot_name+"_loss_all")
        #     plot(rmse_all[k], dataset[k]+"_"+args.plot_name+"_rmse_all")
        #     plot(mae_all[k], dataset[k]+"_"+args.plot_name+"_mae_all")
    