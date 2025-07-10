from model_version_02 import VisionTransformer

model = VisionTransformer(
    img_size=(360, 640),
    patch_size=16,
    embed_dim=1024,
    depth=5,
    num_heads=8,
    mlp_dim=4096,
)
