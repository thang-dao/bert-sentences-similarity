import warnings
warnings.filterwarnings('ignore')

import io
import random
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from bert import data, model
from gluonnlp.data.dataset import SimpleDataset
np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
# change `ctx` to `mx.cpu()` if no GPU is available.
ctx = mx.gpu(3)

class MyDataset(SimpleDataset):
    def __init__(self, sts1, sts2, labels):
        super().__init__(self._read(sts1, sts2, labels))

    def _read(self, sts1, sts2, labels):
        all_samples = []
        for s1, s2, l in zip(sts1, sts2, labels):
            all_samples.append([s1, s2, l])
        return all_samples


bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='wiki_multilingual_cased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
print(bert_base)

bert_classifier = model.classification.BERTClassifier(bert_base, num_classes=2, dropout=0.1)
# only need to initialize the classifier layer.
bert_classifier.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
bert_classifier.hybridize(static_alloc=True)

# softmax cross entropy loss for classification
loss_function = mx.gluon.loss.SoftmaxCELoss()
loss_function.hybridize(static_alloc=True)

metric = mx.metric.Accuracy()

with io.open("Sentences1.txt", "r", encoding="utf8") as f:
    sts1 = f.readlines()

sts1 = [i[:-1] for i in sts1]


with io.open("Sentences2.txt", "r", encoding="utf8") as f:
    sts2 = f.readlines()

sts2 = [i[:-1] for i in sts2]

with open("Labels.txt") as f:
    labels = f.readlines()

labels = [i[0] for i in labels]

# Skip the first line, which is the schema
# num_discard_samples = 1
# # Split fields by tabs
# field_separator = nlp.data.Splitter('\t')
# # Fields to select from the file
# field_indices = [3, 4, 0]

data_train_raw = []


data_train_raw = MyDataset(sts1, sts2, labels)
sample_id = 0
# Sentence A
print(data_train_raw[sample_id][0])
# Sentence B
print(data_train_raw[sample_id][1])
# 1 means equivalent, 0 means not equivalent
print(data_train_raw[sample_id][2])

# Use the vocabulary from pre-trained model for tokenization
bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)

# The maximum length of an input sequence
max_len = 128

# The labels for the two classes [(0 = not similar) or  (1 = similar)]
all_labels = ["0", "1"]

# whether to transform the data as sentence pairs.
# for single sentence classification, set pair=False
# for regression task, set class_labels=None
# for inference without label available, set has_label=False
pair = True
transform = data.transform.BERTDatasetTransform(bert_tokenizer, max_len,
                                                class_labels=all_labels,
                                                has_label=True,
                                                pad=True,
                                                pair=pair)
data_train = data_train_raw.transform(transform)

print('vocabulary used for tokenization = \n%s'%vocabulary)
print('%s token id = %s'%(vocabulary.padding_token, vocabulary[vocabulary.padding_token]))
print('%s token id = %s'%(vocabulary.cls_token, vocabulary[vocabulary.cls_token]))
print('%s token id = %s'%(vocabulary.sep_token, vocabulary[vocabulary.sep_token]))
print('token ids = \n%s'%data_train[sample_id][0])
print('tokenized = \n %s'%vocabulary[data_train[sample_id][0].tolist()])
print('valid length = \n%s'%data_train[sample_id][1])
print('segment ids = \n%s'%data_train[sample_id][2])
print('label = \n%s'%data_train[sample_id][3])

# The hyperparameters
batch_size = 32
lr = 5e-6

# The FixedBucketSampler and the DataLoader for making the mini-batches
train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[1]) for item in data_train],
                                            batch_size=batch_size,
                                            shuffle=True)
bert_dataloader = mx.gluon.data.DataLoader(data_train, batch_sampler=train_sampler)

trainer = mx.gluon.Trainer(bert_classifier.collect_params(), 'adam',
                           {'learning_rate': lr, 'epsilon': 1e-9})

# Collect all differentiable parameters
# `grad_req == 'null'` indicates no gradients are calculated (e.g. constant parameters)
# The gradients for these params are clipped later
params = [p for p in bert_classifier.collect_params().values() if p.grad_req != 'null']
grad_clip = 1

# Training the model with only three epochs
log_interval = 4
num_epochs = 3
for epoch_id in range(num_epochs):
    metric.reset()
    step_loss = 0
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(bert_dataloader):
        with mx.autograd.record():

            # Load the data to the GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)

            # Forward computation
            out = bert_classifier(token_ids, segment_ids, valid_length.astype('float32'))
            ls = loss_function(out, label).mean()

        # And backwards computation
        ls.backward()

        # Gradient clipping
        trainer.allreduce_grads()
        nlp.utils.clip_grad_global_norm(params, 1)
        trainer.update(1)

        step_loss += ls.asscalar()
        metric.update([label], [out])

        # Printing vital information
        if (batch_id + 1) % (log_interval) == 0:
            print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                         .format(epoch_id, batch_id + 1, len(bert_dataloader),
                                 step_loss / log_interval,
                                 trainer.learning_rate, metric.get()[1]))
            step_loss = 0

bert_classifier.save_parameters("model-last.params")
bert_classifier.export("bert-classifier", epoch=2)
