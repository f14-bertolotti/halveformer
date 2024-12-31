import transformers
import datasets
import torch

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, max_length=128, split="train"):
        self.raw = datasets.load_dataset("dair-ai/emotion", "split")
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        self.raw = self.raw.map(
            lambda x: self.bert_tokenizer(x["text"], padding="max_length", truncation=True, max_length=max_length), 
            batch_size=1000,
            batched=True
        )
        self.data = self.raw[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(batch):
        return {
            "src" : torch.tensor([x["input_ids"     ] for x in batch]),
            "msk" : torch.tensor([x["attention_mask"] for x in batch]),
            "tgt" : torch.tensor([x["label"         ] for x in batch])
        }

if __name__ == "__main__":
    dataset = Dataset()
    print(dataset[0])




