CUDA_VISIBLE_DEVICES=2 python3 -u main.py --header_type Linear --chunk_size 1024 --chunk_batch 64 --device 0 --mode testing --missing_mechanism MNAR --load_model_name Linear_Epoch3999_allinone_v3


CUDA_VISIBLE_DEVICES=2,3 nohup python3 -u main.py --header_type LLM --chunk_size 6 --chunk_batch 1 --device 0 --mode testing --missing_mechanism MCAR --load_model_name LLM_Epoch119_allinone_llm_v1 --load_emb --llm_path ../models_hf/llama2_7b/ >> ./logs/UnIMP_LLM_v1.txt 2>&1 &


CUDA_VISIBLE_DEVICES=2,3 nohup python3 -u main.py --header_type LLM --chunk_size 6 --chunk_batch 1 --device 0 --mode testing --missing_mechanism MCAR --load_model_name LLM_Epoch29_allinone_llm_v3 --load_emb --llm_path ../models_hf/llama2_7b/ >> ./logs/UnIMP_LLM_v2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 nohup python3 -u main.py --header_type LLM --chunk_size 6 --chunk_batch 1 --device 0 --mode testing --missing_mechanism MCAR --load_model_name LLM_Epoch59_allinone_llm_v3 --load_emb --llm_path ../models_hf/llama2_7b/ >> ./logs/UnIMP_LLM_v3.txt 2>&1 &


CUDA_VISIBLE_DEVICES=2,3 nohup python3 -u main.py --header_type LLM --chunk_size 6 --chunk_batch 1 --device 0 --mode testing --missing_mechanism MCAR --load_model_name LLM_Epoch299_allinone_llm_v3 --load_emb --llm_path ../models_hf/llama2_7b/ >> ./logs/UnIMP_LLM_v5.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 -u main.py --header_type Linear --chunk_size 1024 --chunk_batch 64 --device 0 --mode testing --missing_mechanism MCAR --load_model_name Linear_Epoch3999_allinone_v3


CUDA_VISIBLE_DEVICES=2,3 nohup python3 -u main.py --header_type LLM --chunk_size 6 --chunk_batch 1 --device 0 --mode testing --missing_mechanism MCAR --load_model_name LLM_Epoch89_allinone_llm_v3 --llm_path ../models_hf/llama2_7b/ >> ./logs/UnIMP_LLM_testing_inductive.txt 2>&1 &