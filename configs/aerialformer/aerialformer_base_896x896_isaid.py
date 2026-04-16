_base_ = ["./aerialformer_tiny_896x896_isaid.py"]
checkpoint_file = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth"  # noqa
decoder_norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=128,
        window_size=12,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        conv_norm_cfg=decoder_norm_cfg
        ),
    decode_head=dict(
        in_channels=[64, 128, 256, 512, 1024],
        channels=128,
        norm_cfg=decoder_norm_cfg,
    )
)

data = dict(samples_per_gpu=8, workers_per_gpu=4) #<---- 1GPU, change at cli call

optimizer_config = dict(grad_clip=dict(max_norm=0.35, norm_type=2))
fp16 = dict(loss_scale='dynamic')