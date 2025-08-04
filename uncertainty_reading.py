import torch
import torch.nn.functional as F
base_path = "/home/roar3/variational-3dgs/output/nb1_test"

for i in range(10):
    ckpt_path = f"{base_path}/chkpnt_sample_{i}.pth"
    data = torch.load(ckpt_path, map_location="cpu")
    model_params = data[0] if isinstance(data, tuple) else data

    # If your offsets are stored at index -2 in the tuple/list
    offsets = model_params[-2] if isinstance(model_params, (list, tuple)) else model_params.get('offsets', None)
    print(f"\n=== Offsets for sample {i} ===")
    if offsets is None:
        print("No 'offsets' found in this checkpoint!")
        continue

    print("Offset keys:", offsets.keys())

    for k, v in offsets.items():
        v_tensor = torch.as_tensor(v)
        real_std = F.softplus(v_tensor)
        print(f"{k}: shape={v_tensor.shape}, min_std={real_std.min().item():.4f}, max_std={real_std.max().item():.4f}, mean_std={real_std.mean().item():.4f}, std_of_std={real_std.std().item():.4f}")


