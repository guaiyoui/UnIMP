CUDA_VISIBLE_DEVICES=2,7 nohup python3 -u main.py --epochs 400 --header_type LLM --chunk_size 4 --chunk_batch 1 --device 0 --eval_epoch_gap 20 --relation_type cross_attn --save_name allinone_llm_v1 --llm_path ../models_hf/llama2_7b/ >> ./logs/All_LLM_v2.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4,6 nohup python3 -u main.py --epochs 400 --header_type LLM --chunk_size 8 --chunk_batch 1 --device 0 --eval_epoch_gap 20 --relation_type cross_attn --save_name allinone_llm_v1 --llm_path ../models_hf/llama2_7b/ --save_emb >> ./logs/All_LLM_v2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=4,6 python3 -u main.py --epochs 400 --header_type LLM --chunk_size 8 --chunk_batch 1 --device 0 --eval_epoch_gap 20 --relation_type cross_attn --save_name allinone_llm_v1 --llm_path ../models_hf/llama2_7b/ --load_emb --mode testing --load_model_name LLM_Epoch19_allinone_llm_v1

CUDA_VISIBLE_DEVICES=4,6 python3 -u main.py --epochs 400 --header_type LLM --chunk_size 8 --chunk_batch 1 --device 0 --eval_epoch_gap 20 --relation_type cross_attn --save_name allinone_llm_v1 --llm_path ../models_hf/llama2_7b/ --load_emb --mode training --load_model_name LLM_Epoch19_allinone_llm_v1


CUDA_VISIBLE_DEVICES=4,6 python3 -u main.py --epochs 400 --header_type LLM --chunk_size 6 --chunk_batch 1 --device 0 --eval_epoch_gap 20 --relation_type cross_attn --save_name allinone_llm_v1 --llm_path ../models_hf/llama2_7b/ --mode training

# non layernorm
CUDA_VISIBLE_DEVICES=4,6 nohup python3 -u main.py --epochs 300 --header_type LLM --chunk_size 6 --chunk_batch 1 --device 0 --eval_epoch_gap 15 --relation_type cross_attn --save_name allinone_llm_v1 --llm_path ../models_hf/llama2_7b/ --mode training --save_emb  >> ./logs/All_LLM_v3.txt 2>&1 &

# have layernorm
CUDA_VISIBLE_DEVICES=5,7 nohup python3 -u main.py --epochs 300 --header_type LLM --chunk_size 6 --chunk_batch 1 --device 0 --eval_epoch_gap 15 --relation_type cross_attn --save_name allinone_llm_v3 --llm_path ../models_hf/llama2_7b/ --mode training --load_emb  >> ./logs/All_LLM_v4_layernorm.txt 2>&1 &