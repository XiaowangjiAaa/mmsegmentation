custom_imports = {'imports': 'mmdet.models', 'allow_failed_imports': False}

model = {
    'backbone': {
        'deep_stem': False,
        'depth': 50,
        'frozen_stages': -1,
        'init_cfg': {
            'checkpoint': 'torchvision://resnet50',
            'type': 'Pretrained'
        },
        'norm_cfg': {
            'requires_grad': False,
            'type': 'SyncBN'
        },
        'num_stages': 4,
        'out_indices': (0, 1, 2, 3),
        'style': 'pytorch',
        'type': 'ResNet'
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
        'enforce_decoder_input_project': False,
        'feat_channels': 256,
        'in_channels': [256, 512, 1024, 2048],
        'loss_cls': {
            'class_weight': [1.0, 1.0, 0.1],
            'loss_weight': 1.0,
            'reduction': 'mean',
            'type': 'mmdet.CrossEntropyLoss',
            'use_sigmoid': False
        },
        'loss_dice': {
            'activate': True,
            'eps': 1.0,
            'loss_weight': 1.0,
            'naive_dice': True,
            'reduction': 'mean',
            'type': 'mmdet.DiceLoss',
            'use_sigmoid': True
        },
        'loss_mask': {
            'loss_weight': 0.5,
            'reduction': 'mean',
            'type': 'mmdet.CrossEntropyLoss',
            'use_sigmoid': True
        },
        'num_classes': 2,
        'num_queries': 100,
        'num_transformer_feat_level': 3,
        'out_channels': 256,
        'pixel_decoder': {
            'act_cfg': {
                'type': 'ReLU'
            },
            'encoder': {
                'init_cfg': None,
                'layer_cfg': {
                    'ffn_cfg': {
                        'act_cfg': {
                            'inplace': True,
                            'type': 'ReLU'
                        },
                        'embed_dims': 256,
                        'feedforward_channels': 1024,
                        'ffn_drop': 0.0,
                        'num_fcs': 2
                    },
                    'self_attn_cfg': {
                        'batch_first': True,
                        'dropout': 0.0,
                        'embed_dims': 256,
                        'im2col_step': 64,
                        'init_cfg': None,
                        'norm_cfg': None,
                        'num_heads': 8,
                        'num_levels': 3,
                        'num_points': 4
                    }
                },
                'num_layers': 6
            },
            'init_cfg': None,
            'norm_cfg': {
                'num_groups': 32,
                'type': 'GN'
            },
            'num_outs': 3,
            'positional_encoding': {
                'normalize': True,
                'num_feats': 128
            },
            'type': 'mmdet.MSDeformAttnPixelDecoder'
        },
        'positional_encoding': {
            'normalize': True,
            'num_feats': 128
        },
        'strides': [4, 8, 16, 32],
        'train_cfg': {
            'assigner': {
                'match_costs': [{
                    'type': 'mmdet.ClassificationCost',
                    'weight': 2.0
                }, {
                    'type': 'mmdet.CrossEntropyLossCost',
                    'use_sigmoid': True,
                    'weight': 5.0
                }, {
                    'eps': 1.0,
                    'pred_act': True,
                    'type': 'mmdet.DiceCost',
                    'weight': 5.0
                }],
                'type':
                'mmdet.HungarianAssigner'
            },
            'importance_sample_ratio': 0.75,
            'num_points': 12544,
            'oversample_ratio': 3.0,
            'sampler': {
                'type': 'mmdet.MaskPseudoSampler'
            }
        },
        'transformer_decoder': {
            'init_cfg': None,
            'layer_cfg': {
                'cross_attn_cfg': {
                    'attn_drop': 0.0,
                    'batch_first': True,
                    'dropout_layer': None,
                    'embed_dims': 256,
                    'num_heads': 8,
                    'proj_drop': 0.0
                },
                'ffn_cfg': {
                    'act_cfg': {
                        'inplace': True,
                        'type': 'ReLU'
                    },
                    'add_identity': True,
                    'dropout_layer': None,
                    'embed_dims': 256,
                    'feedforward_channels': 2048,
                    'ffn_drop': 0.0,
                    'num_fcs': 2
                },
                'self_attn_cfg': {
                    'attn_drop': 0.0,
                    'batch_first': True,
                    'dropout_layer': None,
                    'embed_dims': 256,
                    'num_heads': 8,
                    'proj_drop': 0.0
                }
            },
            'num_layers': 9,
            'return_intermediate': True
        },
        'type': 'Mask2FormerHead'
    },
    'test_cfg': {
        'mode': 'whole'
    },
    'train_cfg': {},
    'type': 'EncoderDecoder'
}
