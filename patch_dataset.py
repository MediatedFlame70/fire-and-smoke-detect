path = 'dataset.py'
content = open(path, 'r', encoding='utf-8').read()

old_transforms = """    def _default_transforms(self) -> T.Compose:
        \"\"\"Default image preprocessing pipeline.\"\"\"
        if self.augment:
            return T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])"""

new_transforms = """    def _default_transforms(self) -> T.Compose:
        \"\"\"Default image preprocessing pipeline with advanced hard-negative & color augmentations.\"\"\"
        if self.augment:
            return T.Compose([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.RandomAutocontrast(p=0.25),
                T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.25),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random"),
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])"""

assert old_transforms in content, "old_transforms not found in dataset.py"
content = content.replace(old_transforms, new_transforms)
open(path, 'w', encoding='utf-8').write(content)
print("Updated dataset.py with advanced augmentations!")
