import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

class MiniSpikingYOLO(nn.Module):
    def __init__(self, grid_size=7, num_boxes=1, num_classes=1):
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        
        spike_grad = surrogate.fast_sigmoid(slope=5)
        beta = 0.85 
        
        # Deep network, but with BatchNorm removed to prevent spike distortion
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        )
        
        self.flatten = nn.Flatten()
        flat_size = 128 * 4 * 4 
        output_dim = self.S * self.S * (self.C + 5 * self.B) 
        
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 2048),
            nn.Dropout(0.3), 
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(2048, output_dim)
        )
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
        # Initialize weights to wake the neurons up
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, num_steps):
        utils.reset(self)
        mem_rec = [] 
        
        for step in range(num_steps):
            spk_feat = self.features(x)
            flat = self.flatten(spk_feat)
            cur_out = self.fc(flat)
            _, mem_out = self.lif_out(cur_out) 
            mem_rec.append(mem_out)
            
        mem_rec = torch.stack(mem_rec, dim=0) 
        out_voltage = mem_rec.mean(dim=0) 
        
        # 1. Standard sigmoid (Removes the 99% confidence glitch)
        out_bounded = torch.sigmoid(out_voltage) 
        
        # 2. Reshape to the YOLO grid
        out_reshaped = out_bounded.view(-1, self.S, self.S, self.C + 5 * self.B)
        
        # 3. THE SCALPEL: We clone the tensor to prevent PyTorch gradient errors,
        # and multiply ONLY the Width and Height (indices 4 and 5) by 2.0.
        # This bypasses the SNN voltage cap safely without frying the network.
        final_out = out_reshaped.clone()
        final_out[..., 4:6] = final_out[..., 4:6] * 2.0 
        
        return final_out