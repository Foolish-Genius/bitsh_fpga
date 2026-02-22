import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

class MicroSpikingYOLO(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        
        # 1. Surrogate Gradient: Essential for training SNNs. 
        # It smooths the non-differentiable step function of a spike so we can backpropagate.
        spike_grad = surrogate.fast_sigmoid()
        beta = 0.95 # Neuron membrane decay rate
        
        # 2. Spiking Feature Extractor (Backbone)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, stride=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
        # 3. YOLO Detection Head
        self.flatten = nn.Flatten()
        # Output shape: Grid * Grid * (Classes + 5 * Boxes)
        output_dim = self.S * self.S * (self.C + 5 * self.B)
        
        # Assuming a 64x64 input image for this micro architecture
        self.fc = nn.Linear(32 * 16 * 16, output_dim) 
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x, num_steps):
        spk_rec = []
        
        # Reset hidden states (membrane potentials) for the new video frame/image
        utils.reset(self)
        
        # SNNs compute over time. We simulate this by looping over 'num_steps'
        for step in range(num_steps):
            # Forward pass through spiking layers
            cur1 = self.conv1(x)
            spk1 = self.lif1(cur1)
            
            cur2 = self.conv2(spk1)
            spk2 = self.lif2(cur2)
            
            flat = self.flatten(spk2)
            cur_out = self.fc(flat)
            spk_out, mem_out = self.lif_out(cur_out)
            
            # Record the output spikes
            spk_rec.append(spk_out)
            
        # Stack spikes over time: Shape -> [Steps, Batch, OutputDim]
        spk_rec = torch.stack(spk_rec, dim=0) 
        
        # YOLO requires continuous numbers for bounding boxes. 
        # We achieve this by calculating the firing rate (mean) of the output spikes over time.
        out_rates = spk_rec.mean(dim=0) 
        
        # Reshape to standard YOLO format: [Batch, S, S, C + 5*B]
        return out_rates.view(-1, self.S, self.S, self.C + 5 * self.B)

def train_spiking_yolo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MicroSpikingYOLO().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Note: A full YOLO uses a complex custom loss function (combining MSE for boxes and CrossEntropy for classes).
    # We use MSE here purely to demonstrate the SNN backpropagation loop.
    criterion = nn.MSELoss() 
    
    num_steps = 10 # Temporal simulation steps
    epochs = 5
    
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        
        # Dummy Data representing PASCAL VOC/COCO: Batch of 8, 3 channels, 64x64 resolution
        inputs = torch.randn(8, 3, 64, 64).to(device) 
        # Dummy Targets: Batch of 8, 7x7 grid, 30 values per grid cell
        targets = torch.rand(8, 7, 7, 30).to(device) 
        
        optimizer.zero_grad()
        
        # Forward pass (executes the temporal loop inside the model)
        outputs = model(inputs, num_steps)
        
        # Loss calculation on the firing rates, then Backpropagation Through Time (BPTT)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_spiking_yolo()