import pandas as pd

labels = []
messages = []

with open("spam.csv", encoding="latin-1") as f:
    for line in f:
        parts = line.split(',', 1)   # split only first comma
        
        if len(parts) < 2:
            continue
        
        label = parts[0].strip().lower()
        message = parts[1].strip()
        
        if label in ['ham', 'spam']:
            labels.append(label)
            messages.append(message)

data = pd.DataFrame({
    'label': labels,
    'message': messages
})

print("Rows:", len(data))

data.to_csv("clean_spam.csv", index=False)