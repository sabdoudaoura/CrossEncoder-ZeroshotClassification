import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossEncoderModel(nn.Module):

    def __init__(self, model_name, max_num_labels):
        """
        Args:
            model_name (str): Name of the pretrained model (e.g., "bert-base-uncased").
            max_num_labels (int): Maximum number of labels that any text can have.
        """
        super(CrossEncoderModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
        self.max_num_labels = max_num_labels

    def _build_crossencoder_inputs(self, text, labels):
        """
        Build a single cross-encoder input for one example:
            [CLS] label_1 [CLS] label_2 ... [CLS] label_k [SEP] text [SEP]

        Returns:
            input_ids (torch.LongTensor): Token indices for the concatenated sequence.
            attention_mask (torch.LongTensor): Attention mask for that sequence.
            label_cls_positions (torch.LongTensor): Indices of the `[CLS]` tokens corresponding to each label.
        """

        labels_concatenated = " ".join(f"{self.tokenizer.cls_token} {label}" for label in labels)
        combined_str = " ".join([labels_concatenated, f"{self.tokenizer.sep_token} {text} {self.tokenizer.sep_token}"])
        encoding = self.tokenizer(combined_str, add_special_tokens=False, padding=True, return_tensors='pt')

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        all_cls_positions = (input_ids == self.tokenizer.cls_token_id).nonzero(as_tuple=True)[1]
        label_cls_positions = all_cls_positions[:len(labels)]

        return input_ids, attention_mask, label_cls_positions

    def forward(self, inputs, device = device):
        """**Forward pass**

        Args:
            inputs (dict[torch.Tensor]): A dictionnary with 3 tensors
            device (str, optional): gpu or cpu. Defaults to device.

        Returns:
            scores: _description_
            mask_labels: 
        """
       
        batch_size = inputs["input_ids"].shape[0]
        max_num_labels = self.max_num_labels
        hidden_dim = self.encoder.config.hidden_size

        label_cls_positions = inputs["cls_position"]
        inputs.pop("cls_position")
        outputs = self.encoder(**inputs)


        # cls_embeddings : [batch_size, max_num_labels, hidden_dim]
        # last_hidden_state : [batch_size, xxx, hidden_dim]
        # mask_labels : [batch_size, max_num_labels]
        cls_embeddings = torch.zeros(batch_size, max_num_labels, hidden_dim).to(device)
        last_hidden_state = outputs.last_hidden_state
        mask_labels = torch.zeros(batch_size, max_num_labels).to(device)
        for i in range(batch_size):
            count = label_cls_positions[i].shape[0]
            if count>0:
                cls_embeddings[i, :count, :] = last_hidden_state[i, label_cls_positions[i].tolist(), :]
                mask_labels[i, :count] = 1

        # scores : [batch_size, max_num_labels]
        scores = self.classifier(cls_embeddings).squeeze(2)
        return scores, mask_labels

    @torch.no_grad()
    def forward_predict(self, texts, labels, device = device):
        """
        Args:
            texts (List[str]): List of input texts.
            labels (List[List[str]]): List of labels corresponding to each text.

        Returns:
            A list of dictionaries, each containing:
               {
                   "text": str,
                   "scores": { label: float_score, ... }
               }
        """

        B = len(texts)
        max_num_labels = self.max_num_labels
        hidden_dim = self.encoder.config.hidden_size

        input_ids, att_mask, label_pos = [], [], []
        for i in range(B):
            input_ids_i, att_mask_i, label_pos_i = self._build_crossencoder_inputs(texts[i], batch_labels[i])

            input_ids.append(input_ids_i.squeeze(0))
            att_mask.append(att_mask_i.squeeze(0))
            label_pos.append(label_pos_i.squeeze(0))
        inputs = self.tokenizer.pad({"input_ids" : input_ids, "attention_mask" : att_mask}, padding=True)
        inputs["cls_position"] = label_pos

        scores, masks = self.forward(inputs, device)
        scores = torch.sigmoid(scores)
        results = []
        for i, text in enumerate(texts):
            text_result = {}
            for j, label in enumerate(labels[i]):
                if masks[i, j]:
                    text_result[label] = float(f"{scores[i, j].item():.2f}")
            results.append({"text": text, "scores": text_result})
        return results