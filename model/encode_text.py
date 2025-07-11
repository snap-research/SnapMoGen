import torch
from transformers import AutoTokenizer, T5EncoderModel


class T5TextEncoder:
    def __init__(self, 
                 device, 
                #  use_text_preprocessing,
                 local_files_only, 
                 from_pretrained=None, 
                 model_max_length=120):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            local_files_only = local_files_only,
            legacy=False
        )
        
        self.model = T5EncoderModel.from_pretrained(
            from_pretrained,
            local_files_only=local_files_only
        ).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # self.use_text_processing = use_text_preprocessing
        self.model_max_length = model_max_length
        # self.tokenizer.to(self.device)
        self.model.to(self.device)

    @torch.no_grad()
    def get_text_embeddings(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = text_tokens_and_mask["input_ids"].to(self.device)
        attention_mask = text_tokens_and_mask["attention_mask"].to(self.device)
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )["last_hidden_state"].detach()
        return text_encoder_embs, attention_mask

