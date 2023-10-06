checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/workspace/PX/adult_children_recognition/mmdetection/pretrained/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
resume_from = None
workflow = [('train', 1)]
no_validate = True