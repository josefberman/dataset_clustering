import pandas as pd
import random

# Configuration
NUM_RECORDS = 100000
OUTPUT_FILE = "dirty_hardware_data.csv"

# Base Data Templates
hardware_templates = {
    "Router": {"models": ["Cisco ISR", "TP-Link Archer", "Netgear Nighthawk"], "sub": ["4331/K9", "AX55", "R7000", "v2"]},
    "Cable": {"models": ["Ethernet Cat6", "HDMI 2.1", "USB-C to Lightning"], "sub": ["50ft", "2m", "Blue", "Braided"]},
    "Webcam": {"models": ["Logitech C920", "Razer Kiyo", "Microsoft LifeCam"], "sub": ["HD Pro", "4K", "Studio"]},
    "Mobile Phone": {"models": ["iPhone 15", "Samsung Galaxy S24", "Pixel 8"], "sub": ["Pro Max", "Ultra", "128GB", "Unlocked"]},
    "SIM Card": {"models": ["Nano SIM", "eSIM", "Micro SIM"], "sub": ["Verizon", "AT&T", "Prepaid", "5G"]}
}

# Chaos Functions
def corrupt_text(text):
    if not text or random.random() > 0.3: return text
    chars = list(text)
    choice = random.random()
    if choice < 0.4: # Typo
        idx = random.randint(0, len(chars) - 1)
        chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
    elif choice < 0.7: # Case swap
        return text.swapcase()
    else: # Shorthand
        return text[:3].lower()
    return "".join(chars)

data = []

for _ in range(NUM_RECORDS):
    # Pick a random category
    category = random.choice(list(hardware_templates.keys()))
    model = random.choice(hardware_templates[category]["models"])
    submodel = random.choice(hardware_templates[category]["sub"])
    
    # Randomly apply corruption
    row = [
        corrupt_text(category),
        corrupt_text(model),
        corrupt_text(submodel)
    ]
    
    # 5% chance of field mismatch (shuffling the data in the row)
    if random.random() < 0.05:
        random.shuffle(row)
        
    # 2% chance of "Null" strings or garbage
    if random.random() < 0.02:
        row[random.randint(0, 2)] = random.choice(["n/a", "???", "---", "unknown", "0"])

    data.append(row)

# Create DataFrame and Save
df = pd.DataFrame(data, columns=["Type of hardware", "Model", "Submodel"])
df.to_csv(OUTPUT_FILE, index=False)

print(f"Successfully generated {NUM_RECORDS} dirty records in {OUTPUT_FILE}")