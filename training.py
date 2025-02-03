import torch
from torch import nn
import json
import yaml
from model import CrossEncoderModel
from torch.optim import AdamW
from dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score  
from sklearn.model_selection import train_test_split

# Load the configuration file
with open("\config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access parameters
train_data_path = config["data"]["data_path"]
model_name = config["model"]["name"]
max_num_labels = config["model"]["max_num_labels"]
learning_rate = float(config["training"]["learning_rate"])
batch_size = config["training"]["batch_size"]

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = config["training"]["epochs"]

tokenizer = AutoTokenizer.from_pretrained(model_name)

#Custom collate function
def custom_collate_fn(batch):
    # Separate texts and labels from the batch
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask" ] for item in batch]
    cls_position = [item["cls_position"] for item in batch]
    target_labels = [item["target_labels"] for item in batch]
    target_labels = torch.stack(target_labels, dim=0)
    inputs = tokenizer.pad({"input_ids" : input_ids, "attention_mask" : attention_mask}, padding=True)
    inputs["cls_position"] = cls_position
    return inputs, target_labels

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, target_labels in dataloader:
            # Ensure inputs are moved to device (handle dictionaries appropriately)
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            target_labels = target_labels.to(device)

            scores, mask = model.forward(inputs, target_labels)
            predictions = (torch.sigmoid(scores) > 0.5).float()

            valid_preds = predictions[mask.bool()]
            valid_targets = target_labels[mask.bool()]

            all_preds.append(valid_preds.cpu())
            all_targets.append(valid_targets.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    # Compute F1 score
    f1 = f1_score(all_targets, all_preds, average="micro")
    return f1




if __name__ == "__main__":

    model = CrossEncoderModel(model_name, max_num_labels).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none') # multiclass classification, reduction to none due to padding
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    

    with open(train_data_path, 'r', encoding='utf-8') as f:
      train_data = json.load(f)

    # Split train and validation
    train_set, validation_set = train_test_split(train_data, test_size=0.1, random_state=42, shuffle=True)

    # Datasets
    train_dataset = CustomDataset(train_set, model_name, max_num_labels)
    validation_dataset = CustomDataset(validation_set, model_name, max_num_labels)

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    validation_scores = []
    loss_scores = []
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for batch_idx, (inputs, target_labels) in enumerate(train_loader):

            inputs = inputs.to(device)
            target_labels = target_labels.to(device)

            #Forward pass
            optimizer.zero_grad()

            # scores : [batch_size, max_num_labels]
            # mask : [batch_size, max_num_labels]
            scores, mask = model.forward(inputs, device)

            # Compute the loss
            loss = criterion(scores, target_labels)

            # Mask
            loss = (loss * mask).sum()

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            # Log de la perte
            running_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / (batch_idx + 1):.4f}")
        
          
        ## Evaluation
        f1 = evaluate(model, validation_loader, device) 
        print(f"Epoch [{epoch + 1}/{epochs}] Evaluation F1 Score: {f1:.4f}")
        loss_scores.append(running_loss / (batch_idx + 1))
        validation_scores.append(f1)
        