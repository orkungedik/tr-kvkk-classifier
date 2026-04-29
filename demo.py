import torch
import torch.nn as nn
import json
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

REPO_ID = "orkungedik/tr-kvkk-classification"  # Hugging Face repo ID
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharViT(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, n_heads, n_layers, num_classes):
        super(CharViT, self).__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True, dropout=0.2, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, input_ids):
        b, seq_len = input_ids.shape

        pad_mask = (input_ids == 0)
        cls_mask = torch.zeros((b, 1), dtype=torch.bool, device=input_ids.device)
        full_mask = torch.cat((cls_mask, pad_mask), dim=1)

        x          = self.embed(input_ids)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x          = torch.cat((cls_tokens, x), dim=1)
        x          = x + self.pos_embed[:, :seq_len + 1, :]  # dinamik slice

        x = self.transformer(x, src_key_padding_mask=full_mask)
        return self.classifier(x[:, 0])

# Dosyaları indir
model_path  = hf_hub_download("orkungedik/tr-kvkk-classifier", "model.safetensors")
vocab_path  = hf_hub_download("orkungedik/tr-kvkk-classifier", "vocab.json")
config_path = hf_hub_download("orkungedik/tr-kvkk-classifier", "config.json")

with open(vocab_path)  as f: char2idx = json.load(f)
with open(config_path) as f: config   = json.load(f)

model = CharViT(**{k: config[k] for k in
    ["vocab_size","embed_dim","max_len","n_heads","n_layers","num_classes"]})
model.load_state_dict(load_file(model_path))
model.eval()

def predict(text: str) -> str:
    unk = char2idx["[UNK]"]
    ids = [char2idx.get(c, unk) for c in text[:config["max_len"]]]
    ids += [0] * (config["max_len"] - len(ids))
    with torch.no_grad():
        logits = model(torch.tensor([ids]))
    return config["label_names"][logits.argmax().item()]

print(predict("TC kimlik: 12345678901"))
