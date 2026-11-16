"""Read the cutoff and the number of message-passing layers of MACE-MP-0 small (for the receptive field in 10.1)."""
import os, warnings; warnings.filterwarnings('ignore')
import torch; torch.set_num_threads(1)
m = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'mace_mp0_small.model'),
               map_location='cpu', weights_only=False)
rc = float(m.r_max); T = int(m.num_interactions)
print(f"MACE-MP-0 small: r_max = {rc:.1f} A, interaction layers = {T}, receptive field T*r_max = {T*rc:.0f} A")
print(f"parameters: {sum(p.numel() for p in m.parameters())}")
