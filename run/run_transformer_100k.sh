CUDA_VISIBLE_DEVICES=1 python main_from_args.py ./args/transformer4ea_args_100K.json EN_DE_100K_V1 721_5fold/1/ > en_de_100k

CUDA_VISIBLE_DEVICES=1 python main_from_args.py ./args/transformer4ea_args_100K.json EN_FR_100K_V1 721_5fold/1/ > en_fr_100k

CUDA_VISIBLE_DEVICES=1 python main_from_args.py ./args/transformer4ea_args_100K.json D_W_100K_V1 721_5fold/1/ > d_w_100k

CUDA_VISIBLE_DEVICES=1 python main_from_args.py ./args/transformer4ea_args_100K.json D_Y_100K_V1 721_5fold/1/ > d_y_100k
