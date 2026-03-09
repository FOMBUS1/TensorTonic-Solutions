import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id = {self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3}
        self.id_to_word = {0: self.pad_token, 1: self.unk_token, 2: self.bos_token, 3: self.eos_token}

        word_idx = 4
        
        words = set(word for text in texts for word in text.split(' '))
        for word in words:
            if not(self.word_to_id.get(word)):
                self.word_to_id[word] = word_idx
                self.id_to_word[word_idx] = word

                word_idx += 1

        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        words = text.split(' ')
        encoded_text = []
        for word in words:
            if self.word_to_id.get(word):
                encoded_text.append(self.word_to_id[word])
            else:
                encoded_text.append(self.word_to_id[self.unk_token])
                
        return encoded_text
            
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        decoded_text = []
        for idx in ids:
            decoded_text.append(self.id_to_word[idx])
            
        return " ".join(decoded_text).strip()