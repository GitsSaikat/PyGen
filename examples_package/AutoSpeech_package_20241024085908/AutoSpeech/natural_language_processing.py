import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List, Dict, Any

class EnhancedTextPreprocessing:
    def __init__(self, language: str = 'english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.logger = logging.getLogger(__name__)
        
        # Initialize BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove special characters
            text = re.sub(r'[^\w\s]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text
        except Exception as e:
            self.logger.error(f"Error in text cleaning: {str(e)}")
            raise

    def lemmatize_text(self, tokens: List[str]) -> List[str]:
        """Lemmatization with POS tagging"""
        try:
            pos_tags = nltk.pos_tag(tokens)
            return [self.lemmatizer.lemmatize(word, self._get_wordnet_pos(tag))
                   for word, tag in pos_tags]
        except Exception as e:
            self.logger.error(f"Error in lemmatization: {str(e)}")
            raise

    def _get_wordnet_pos(self, tag: str) -> str:
        """Map POS tag to WordNet POS tag"""
        tag_dict = {
            'J': nltk.corpus.wordnet.ADJ,
            'N': nltk.corpus.wordnet.NOUN,
            'V': nltk.corpus.wordnet.VERB,
            'R': nltk.corpus.wordnet.ADV
        }
        return tag_dict.get(tag[0], nltk.corpus.wordnet.NOUN)

class TransformerClassifier(nn.Module):
    """Enhanced text classification using BERT"""
    
    def __init__(self, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                          attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class EnhancedTextDataset(Dataset):
    """Enhanced dataset with BERT tokenization"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer: AutoTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Example usage
if __name__ == '__main__':
    # Initialize text preprocessing object
    text_preprocessing = EnhancedTextPreprocessing()

    # Example text
    text = "This is an example sentence for text preprocessing."
    preprocessed_text = text_preprocessing.clean_text(text)
    print(preprocessed_text)

    # Create dataset
    dataset = EnhancedTextDataset([preprocessed_text], [1])  # Assuming text is labeled as 1

    # Create data loader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize text classification model
    model = TransformerClassifier(num_classes=2, dropout=0.1)

    # Train the model
    for batch in data_loader:
        input_ids, attention_mask, label = batch
        out = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(out, label)
        print(loss.item())
