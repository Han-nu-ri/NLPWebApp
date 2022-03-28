import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np

#kobert
import pytorch_kobert


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=6,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


class SentimentAnalyzer():
    def __init__(self):
        self.emotion_to_int = {'기쁨': 0, '불안': 1, '당황': 2, '슬픔': 3, '분노': 4, '상처': 5}
        self.int_to_emotion = {0: '기쁨', 1: '불안', 2: '당황', 3: '슬픔', 4: '분노', 5: '상처'}
        bertmodel, vocab = pytorch_kobert.get_pytorch_kobert_model()
        tokenizer = pytorch_kobert.get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
        self.device = torch.device("cpu")
        self.model = BERTClassifier(bertmodel, dr_rate=0.5).to(self.device)
        self.model.load_state_dict(torch.load('./model_weight/kobert_state_dict_30.pth', map_location=self.device))

    def predict(self, predict_sentence, max_len=64, batch_size=64):
        test_X = []
        for each_sentence in predict_sentence:
            test_X.append([each_sentence, '0'])
        test_BERT_X = BERTDataset(test_X, 0, 1, self.tok, max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(test_BERT_X, batch_size=batch_size)

        self.model.eval()
        test_eval = []
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            valid_length = valid_length
            label = label.long().to(self.device)

            Y_hat = self.model(token_ids, valid_length, segment_ids)
            for i in Y_hat:
                logits = i
                logits = logits.detach().cpu().numpy()
                test_eval.append(np.argmax(logits))
        return np.array([self.int_to_emotion[each_element] for each_element in test_eval])

sentiment_analyzer = SentimentAnalyzer()
def lambda_handler(event, context):
    # extract values from the event object we got from the Lambda service
    target_text = event['target_text']
    sentiment_result = sentiment_analyzer.predict([target_text])[0]
    # return a properly formatted JSON object
    return {
        'statusCode': 200,
        'sentiment_result': sentiment_result
    }
