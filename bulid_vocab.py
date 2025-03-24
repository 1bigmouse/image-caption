import pickle
from collections import Counter
import nltk
from pycocotools.coco import COCO


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()

    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    words = [word for word, cnt in counter.items() if cnt >= threshold]   # 次数大于阈值的就留下来

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(caption_path, threshold, save_path):
    vocab = build_vocab(json=caption_path, threshold=threshold)
    vocab_path = save_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Total vocabulary size: {len(vocab)}")
    print(f"Saved the vocabulary wrapper to '{vocab_path}'")

if __name__ == "__main__":
    args = {"caption_path": r"D:\Extend E Python project\DL_design\COCO\annotations\captions_train2014.json",
            "threshold": 4,
            "save_path": r"D:\Extend E Python project\DL_design\pkl\vocab.pkl"}
    main(**args)
    