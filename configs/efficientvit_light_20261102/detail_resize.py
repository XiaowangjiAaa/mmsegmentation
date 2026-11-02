# Isolate the effect of two-stage bilinear resizing from learned detail.
_base_ = ['./_base_.py']
model = dict(decode_head=dict(detail_upsample_only=True, detail_in_index=0))
