path = 'train.py'
content = open(path, 'r', encoding='utf-8').read()

old_weighting = """    counts = torch.zeros(4, dtype=torch.float32)
    for sample in train_loader.dataset.samples:
        label = train_loader.dataset.sample_class_label(sample)
        counts[label] += 1
    class_weights = (counts.sum() / (4.0 * counts.clamp_min(1))).clamp(max=5.0).to(device)
    print(f"Training class counts: {counts.int().tolist()}")
    print(f"Class loss weights:    {class_weights.cpu().tolist()}")"""

new_weighting = """    counts = torch.zeros(4, dtype=torch.float32)
    for sample in train_loader.dataset.samples:
        label = train_loader.dataset.sample_class_label(sample)
        counts[label] += 1
    # Normalized square-root dampened inverse frequency weighting for 4-way classification
    raw_weights = (counts.sum() / (4.0 * counts.clamp_min(1))).pow(0.5)
    class_weights = (raw_weights / raw_weights.mean()).clamp(min=0.5, max=3.5).to(device)
    print(f"Training class counts: {counts.int().tolist()}")
    print(f"Class loss weights:    {class_weights.cpu().tolist()}")"""

assert old_weighting in content, "old_weighting not found"
content = content.replace(old_weighting, new_weighting)

# Update default bbox-weight to 1.0, weight-decay to 0.01, epochs to 25, patience to 8
content = content.replace('parser.add_argument("--bbox-weight", type=float, default=1.5,',
                          'parser.add_argument("--bbox-weight", type=float, default=1.0,')
content = content.replace('parser.add_argument("--weight-decay", type=float, default=0.05,',
                          'parser.add_argument("--weight-decay", type=float, default=0.01,')
content = content.replace('parser.add_argument("--epochs", type=int, default=20,',
                          'parser.add_argument("--epochs", type=int, default=25,')
content = content.replace('parser.add_argument("--patience", type=int, default=15,',
                          'parser.add_argument("--patience", type=int, default=8,')

open(path, 'w', encoding='utf-8').write(content)
print("Updated train.py parameters for optimal 4-way classification accuracy!")
