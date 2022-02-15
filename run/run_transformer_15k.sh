CUDA_VISIBLE_DEVICES=0 python main_from_args.py ./args/transformer4ea_args_15K.json D_W_15K_V1 721_5fold/1/ > d_w_15k

CUDA_VISIBLE_DEVICES=0 python main_from_args.py ./args/transformer4ea_args_15K.json EN_FR_15K_V1 721_5fold/1/ > en_fr_15k

CUDA_VISIBLE_DEVICES=0 python main_from_args.py ./args/transformer4ea_args_15K.json D_Y_15K_V1 721_5fold/1/ > d_y_15k

CUDA_VISIBLE_DEVICES=0 python main_from_args.py ./args/transformer4ea_args_15K.json EN_DE_15K_V1 721_5fold/1/ > en_de_15k
