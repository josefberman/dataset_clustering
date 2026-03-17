import os
import pandas as pd
import random
import re

# Configuration
NUM_RECORDS = 100000
OUTPUT_FILE = "data/dirty_hardware_data_40k.csv"

# Base Data Templates
hardware_templates = {
    "Router": {"models": ["Cisco ISR", "TP-Link Archer", "Netgear Nighthawk"], "sub": ["4331/K9", "AX55", "R7000", "v2"]},
    "Cable": {"models": ["Ethernet Cat6", "HDMI 2.1", "USB-C to Lightning", "DisplayPort 1.4"], "sub": ["50ft", "2m", "Blue", "Braided"]},
    "Webcam": {"models": ["Logitech C920", "Razer Kiyo", "Microsoft LifeCam", "Elgato Facecam"], "sub": ["HD Pro", "4K", "Studio", "1080p60"]},
    "Mobile Phone": {"models": ["iPhone 15", "Samsung Galaxy S24", "Pixel 8", "OnePlus 12"], "sub": ["Pro Max", "Ultra", "128GB", "Unlocked"]},
    "SIM Card": {"models": ["Nano SIM", "eSIM", "Micro SIM"], "sub": ["Verizon", "AT&T", "Prepaid", "5G"]},
    "Keyboard": {"models": ["Logitech MX Keys", "Keychron Q1", "Razer BlackWidow", "Corsair K70"], "sub": ["Tactile", "Linear", "RGB", "Wireless"]},
    "Mouse": {"models": ["Logitech MX Master 3", "Razer DeathAdder", "SteelSeries Rival 3", "Apple Magic Mouse"], "sub": ["Wireless", "Wired", "Optical", "Ergonomic"]},
    "Headphones": {"models": ["Sony WH-1000XM5", "Bose QuietComfort 45", "Sennheiser HD 600", "AirPods Max"], "sub": ["Noise Cancelling", "Over-Ear", "Wireless", "Studio"]},
    "Monitor": {"models": ["Dell UltraSharp", "LG UltraGear", "ASUS ROG Swift", "Samsung Odyssey G9"], "sub": ["4K", "144Hz", "Ultrawide", "OLED"]},
    "Laptop": {"models": ["MacBook Pro 16", "Dell XPS 15", "Lenovo ThinkPad X1", "Razer Blade 15"], "sub": ["M3 Max", "Core i9", "32GB RAM", "OLED Touch"]}
}

# Advanced Chaos Functions
def corrupt_text(text):
    if not text or random.random() > 0.85: return text # 85% chance to corrupt
    
    choice = random.random()
    if choice < 0.2: # Typo (random character change)
        chars = list(text)
        idx = random.randint(0, len(chars) - 1)
        chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*')
        return "".join(chars)
    elif choice < 0.35: # Case swap
        return text.swapcase()
    elif choice < 0.5: # Shorthand / truncate
        return text[:random.randint(2, 5)].lower()
    elif choice < 0.65: # Remove spaces
        return text.replace(" ", "")
    elif choice < 0.75: # Remove vowels
        return re.sub(r'[aeiouAEIOU]', '', text)
    elif choice < 0.85: # Duplicate random character
        chars = list(text)
        idx = random.randint(0, len(chars) - 1)
        chars.insert(idx, chars[idx])
        return "".join(chars)
    else: # Add random noise prefix/suffix
        noise = "".join(random.choices('0123456789abcxyz', k=3))
        if random.random() > 0.5:
            return f"{noise}_{text}"
        return f"{text}-{noise}"

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
    
    # 25% chance of severe field mismatch (shuffling the data in the row)
    if random.random() < 0.25:
        random.shuffle(row)
        
    # 10% chance of combining two columns into one and leaving one empty
    if random.random() < 0.10:
        idx1, idx2 = random.sample([0, 1, 2], 2)
        row[idx1] = f"{row[idx1]} {row[idx2]}"
        row[idx2] = ""

    # 15% chance of "Null" strings or garbage
    if random.random() < 0.15:
        row[random.randint(0, 2)] = random.choice(["n/a", "???", "---", "unknown", "0", "NULL", "None", "  ", "#REF!"])

    # 5% chance of completely dropping a value
    if random.random() < 0.05:
        row[random.randint(0, 2)] = ""

    data.append(row)

# Create DataFrame and Save
df = pd.DataFrame(data, columns=["Type of hardware", "Model", "Submodel"])
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Successfully generated {NUM_RECORDS} EXTREMELY dirty records in {OUTPUT_FILE}")