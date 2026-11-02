_base_ = ['./_base_.py']
model = dict(decode_head=dict(
    detail_channels=8, detail_in_index=0, detail_in_channels=16))
