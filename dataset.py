import random
import torch
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self, data, model_name, max_num_labels=6, transform=None, max_num_negatives=5):
        """Initializes the CustomDataset object.

        Args:
            data (list): List of dictionaries containing the data. Each dictionary should have the keys "text" and "labels".
            model_name (str): Name of the pre-trained model to use for tokenization.
            max_num_labels (int, optional): Maximum number of labels to consider. Defaults to 6.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
            max_num_negatives (int, optional): Maximum number of negative labels to sample. Defaults to 5.
        """
        self.data = data
        self.max_num_negatives = max_num_negatives
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.all_labels = list(set([label for labels in [element["labels"] for element in data] for label in labels]))
        self.max_num_labels = max_num_labels

    def __getitem__(self, index):
        """Computes the input tensors and target labels for a given index.

        Args:
            index (int): Index of the data point to retrieve.

        Returns:
            dict: A dictionary containing the input tensors, the attention mask, [CLS] positions and target labels.
        """
        negative_labels_list = self.negative_sampling([self.data[index]['labels']], self.all_labels)
        negative_labels = negative_labels_list[0]
        pos_labels = self.data[index]["labels"]

        p = len(pos_labels)
        q = len(negative_labels)
        assert self.max_num_labels >= (p + q), "Maximum number of labels should be >= to p + q"
        
        # Créer une liste de tuples (label, target) pour positives (target=1) et négatives (target=0)
        combined = [(label, 1) for label in pos_labels] + [(label, 0) for label in negative_labels]
        # Mélanger aléatoirement tout en conservant la correspondance entre label et cible
        random.shuffle(combined)
        # Extraire la liste des labels et la liste des cibles
        labels = [elem[0] for elem in combined]
        target_values = [elem[1] for elem in combined]
        target_labels = torch.tensor(target_values, dtype=torch.float)
        # Ajouter un padding de 0 si nécessaire pour atteindre self.max_num_labels
        if len(target_labels) < self.max_num_labels:
            padding = torch.zeros(self.max_num_labels - len(target_labels))
            target_labels = torch.cat([target_labels, padding])
        
        # Construction de la chaîne à tokeniser en préfixant chaque label par le token CLS
        labels_concatenated = " ".join(f"{self.tokenizer.cls_token} {label}" for label in labels)
        text = self.data[index]["text"]
        combined_str = " ".join([
            labels_concatenated,
            f"{self.tokenizer.sep_token} {text} {self.tokenizer.sep_token}"
        ])
        encoding = self.tokenizer(combined_str, add_special_tokens=False, padding=True, return_tensors='pt')

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Récupérer les positions des tokens CLS
        all_cls_positions = (input_ids == self.tokenizer.cls_token_id).nonzero(as_tuple=True)[1]
        label_cls_positions = all_cls_positions[:len(labels)]

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "cls_position": label_cls_positions.squeeze(0),
            "target_labels": target_labels
        }

    def __len__(self):
        return len(self.data)

    def negative_sampling(self, batch_labels, all_labels):
        """
        Generate negative examples by sampling from all_labels excluding the positive labels.

        Args:
            batch_labels (List[str]): List of lists containing positive labels for each example in the batch.
            all_labels (List[List[str]]): List of all possible labels.

        Returns:
            List[List[str]]: List of lists containing negative labels for each example in the batch.
        """

        num_negatives = random.randint(1, self.max_num_negatives)
        negative_samples = []
        for labels in batch_labels:
            neg = random.sample([l for l in all_labels if l not in labels], num_negatives)
            negative_samples.append(neg)
        return negative_samples