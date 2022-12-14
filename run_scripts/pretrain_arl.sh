python main.py with data_root=data/pretrain_arrows_umls/ \
 num_gpus=4 num_nodes=1 \
 task_pretrain_arl \
 per_gpu_batchsize=16 \
 clip16 text_roberta \
 image_size=288 max_text_len=64 max_num_ents=24 \
 tokenizer=downloaded/roberta-base \
 load_path=downloaded/meter.ckpt