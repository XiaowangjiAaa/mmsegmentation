model = {
    'backbone': {
        'act_cfg': {
            'type': 'GELU'
        },
        'attention_kernel_paddings': [2, [0, 3], [0, 5], [0, 10]],
        'attention_kernel_sizes': [5, [1, 7], [1, 11], [1, 21]],
        'depths': [3, 3, 5, 2],
        'drop_path_rate': 0.1,
        'drop_rate': 0.0,
        'embed_dims': [32, 64, 160, 256],
        'init_cfg': {
            'checkpoint':
            ('https://download.openmmlab.com/mmsegmentation/v0.5/'
             'pretrain/segnext/mscan_t_20230227-119e8c9f.pth'),
            'type': 'Pretrained'
        },
        'mlp_ratios': [8, 8, 4, 4],
        'norm_cfg': {
            'requires_grad': True,
            'type': 'BN'
        },
        'type': 'MSCAN'
    },
    'data_preprocessor': {
        'bgr_to_rgb': True,
        'mean': [123.675, 116.28, 103.53],
        'pad_val': 0,
        'seg_pad_val': 255,
        'size': (512, 512),
        'std': [58.395, 57.12, 57.375],
        'test_cfg': {
            'size_divisor': 32
        },
        'type': 'SegDataPreProcessor'
    },
    'decode_head': {
        'align_corners': False,
        'channels': 256,
        'dropout_ratio': 0.1,
        'ham_channels': 256,
        'ham_kwargs': {
            'MD_R': 16,
            'MD_S': 1,
            'eval_steps': 7,
            'inv_t': 100,
            'rand_init': True,
            'train_steps': 6
        },
        'in_channels': [64, 160, 256],
        'in_index': [1, 2, 3],
        'loss_decode': [
            {
                'avg_non_ignore': True,
                'loss_weight': 0.5,
                'type': 'CrossEntropyLoss',
                'use_sigmoid': False
            },
            {
                'alpha': 0.3,
                'beta': 0.7,
                'ignore_index': 255,
                'loss_weight': 2.0,
                'type': 'TverskyLoss'
            }
        ],
        'norm_cfg': {
            'num_groups': 32,
            'requires_grad': True,
            'type': 'GN'
        },
        'num_classes': 2,
        'out_channels': 2,
        'type': 'LightHamHead'
    },
    'pretrained': None,
    'test_cfg': {
        'mode': 'whole'
    },
    'train_cfg': {},
    'type': 'EncoderDecoder'
}
