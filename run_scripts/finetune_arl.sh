seed=42
num_gpus=1
per_gpu_batchsize=16
finetune_embeddings=True
load_path=<Path to The Pre-trained Model>

# === VQA ===
python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_vqa_rad \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 finetune_embeddings=${finetune_embeddings} \
 load_path=${load_path}

python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_slack \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 finetune_embeddings=${finetune_embeddings} \
 load_path=${load_path} \
 clip_resizedcrop

python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_medvqa_2019 \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 finetune_embeddings=${finetune_embeddings} \
 load_path=${load_path} \
 clip_resizedcrop

# === CLS ===
python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_cls_melinda_p_meth_label \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 finetune_embeddings=${finetune_embeddings} \
 load_path=${load_path} \
 clip_resizedcrop

# === IRTR ===
python main.py with data_root=data/finetune_arrows_umls/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_irtr_roco get_recall_metric=True \
 pwdper_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=288 \
 tokenizer=downloaded/roberta-base \
 finetune_embeddings=${finetune_embeddings} \
 test_only=True \
 load_path=${load_path}

python main.py with data_root=data/finetune_arrows_umls/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_irtr_roco get_recall_metric=False \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 finetune_embeddings=${finetune_embeddings} \
 load_path=${load_path} \
 clip_resizedcrop
