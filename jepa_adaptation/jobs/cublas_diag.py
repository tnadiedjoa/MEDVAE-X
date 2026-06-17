import torch

print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("driver/runtime:", torch.cuda.get_device_capability(0))


def test(name, fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        print(f"[OK]   {name}  -> {tuple(out.shape)} {out.dtype}")
    except Exception as e:
        print(f"[FAIL] {name}  -> {type(e).__name__}: {str(e)[:160]}")


dev = "cuda"
for dt in (torch.float32, torch.bfloat16):
    a = torch.randn(1024, 1024, device=dev, dtype=dt)
    b = torch.randn(1024, 1024, device=dev, dtype=dt)
    test(f"mm 1024x1024 {dt}", lambda a=a, b=b: a @ b)

# exact failing attention shapes: q[b,hw,c] @ k[b,c,hw]
for dt in (torch.float32, torch.bfloat16):
    q = torch.randn(16, 3136, 512, device=dev, dtype=dt)
    k = torch.randn(16, 512, 3136, device=dev, dtype=dt)
    test(f"bmm contig [16,3136,512]@[16,512,3136] {dt}", lambda q=q, k=k: torch.bmm(q, k))

# transposed (non-contiguous) q, as in the real attention
for dt in (torch.float32, torch.bfloat16):
    qc = torch.randn(16, 512, 3136, device=dev, dtype=dt)
    q = qc.permute(0, 2, 1)  # non-contiguous b,hw,c
    k = torch.randn(16, 512, 3136, device=dev, dtype=dt)
    test(f"bmm transposed-q {dt}", lambda q=q, k=k: torch.bmm(q, k))

# smaller batched to see if it is a size/stride threshold
for hw in (64, 256, 1024, 3136):
    q = torch.randn(16, hw, 512, device=dev, dtype=torch.float32)
    k = torch.randn(16, 512, hw, device=dev, dtype=torch.float32)
    test(f"bmm fp32 hw={hw}", lambda q=q, k=k: torch.bmm(q, k))

print("done")
