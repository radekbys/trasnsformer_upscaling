from model import VisionTransformer

model = VisionTransformer(
    img_size=(360, 640), patch_size=8, embed_dim=512, depth=4, num_heads=8, mlp_dim=1024
)
