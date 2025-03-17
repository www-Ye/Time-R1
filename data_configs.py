DATASETS={
    'charades': {
        'video_path': './Charades/Charades_v1',
        'stride': 20,
        'max_stride_factor': 0.5,
        'splits': {
            'default': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 0.0,
            },
            'small': {
                'annotation_file': 'dataset/charades-sta/charades_test_small.json',
                'pad_sec': 0.0,
            },
            'OOD-1': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 10.0,
            },
            'OOD-2': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 15.0,
            },
            'test-ood': {
                'annotation_file': 'dataset/charades-sta/charades_test_ood.json',
                'pad_sec': 0.0,
            },
            'novel-composition': {
                'annotation_file': 'dataset/charades-sta/novel_composition.json',
                'pad_sec': 0.0,
            },
            'novel-word': {
                'annotation_file': 'dataset/charades-sta/novel_word.json',
                'pad_sec': 0.0,
            },
        }
    },
    'activitynet': {
        'video_path': './activitynet/videos',
        'stride': 40,
        'max_stride_factor': 1,
        'splits': {
            'default': {
                'annotation_file': 'dataset/activitynet/test.json',
                'pad_sec': 0.0,
            },
            'OOD-1': {
                'annotation_file': 'dataset/activitynet/test.json',
                'pad_sec': 30.0,
            },
            'OOD-2': {
                'annotation_file': 'dataset/activitynet/test.json',
                'pad_sec': 60.0,
            },
        }
    },
    'qvhighlight': {
        'feature_path': '/home/zhengmh/Datasets/qvhighlight/blip2_coco_features/',
        'stride': 50,
        'max_stride_factor': 0.5,
        'splits': {
            'default': {
                'annotation_file': 'dataset/qvhighlight/highlight_val_release.jsonl',
                'pad_sec': 0.0,
            },
        }
    },
}