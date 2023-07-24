import transformers as ts
from transformers import DataCollatorForTokenClassification

from datasets import Dataset
import evaluate
import torch

import numpy as np
import pandas as pd
import csv

# datasetName = "NCBI-disease"
# modelPath = "/data/Compact-Biomedical-Transformers/distil-biobert"
# # modelPath = "nlpie/distil-biobert"
# tokenizerPath = modelPath

# datasetPath = f"biobert-datasets/datasets/NER/{datasetName}/"
# logsPath = f"ner_logs/{modelPath}-{datasetName}-logs.txt"

def load_and_preprocess_dataset(datasetPath, tokenizer):
    def load_ner_dataset(folder):
        allLabels = set(pd.read_csv(folder + "train.tsv", sep="\t",
                                    header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')[1])

        label_to_index = {label: index for index,
                          label in enumerate(allLabels)}
        index_to_label = {index: label for index,
                          label in enumerate(allLabels)}

        def load_subset(subset):
            lines = []

            with open(folder + subset, mode="r") as f:
                lines = f.readlines()

            sentences = []
            labels = []

            currentSampleTokens = []
            currentSampleLabels = []

            for line in lines:
                if line.strip() == "":
                    sentences.append(currentSampleTokens)
                    labels.append(currentSampleLabels)
                    currentSampleTokens = []
                    currentSampleLabels = []
                else:
                    cleanedLine = line.replace("\n", "")
                    token, label = cleanedLine.split(
                        "\t")[0].strip(), cleanedLine.split("\t")[1].strip()
                    currentSampleTokens.append(token)
                    currentSampleLabels.append(label_to_index[label])

            dataDict = {
                "tokens": sentences,
                "ner_tags": labels,
            }

            return Dataset.from_dict(dataDict)

        trainingDataset = load_subset("train.tsv")
        # validationDataset = Dataset.from_dict(
        #     load_subset("train_dev.tsv")[len(trainingDataset):])
        validationDataset = Dataset.from_dict(
            load_subset("devel.tsv")[len(trainingDataset):])
        testDataset = load_subset("test.tsv")

        return {
            "train": trainingDataset,
            "validation": validationDataset,
            "test": testDataset,
            "all_ner_tags": list(allLabels),
        }

    dataset = load_ner_dataset(datasetPath)

    label_names = dataset["all_ner_tags"]

    # Get the values for input_ids, token_type_ids, attention_mask
    def tokenize_adjust_labels(all_samples_per_split):

        max_word_length = 50
        max_word_seq_length = 256
        tokenized_sentences = all_samples_per_split['tokens']

        # 将单词分割为字符
        char_seq_list = [[list(word) for word in sentence] for sentence in tokenized_sentences]

        # 填充所有单词，使它们的长度相同
        for sentence in char_seq_list:
            for word in sentence:
                while len(word) < max_word_length:
                    word.append('')

        # 将填充后的单词按照每个句子包含的单词数排列
        word_seq_list = [sentence + [[''] * max_word_length] * (max_word_seq_length - len(sentence)) for sentence in char_seq_list]

        # 将二维列表转换为一个三维张量
        batch_size = len(word_seq_list)
        word_seq_length = max_word_seq_length
        char_seq_length = max_word_length
        input_tensor = torch.zeros((batch_size, word_seq_length, char_seq_length), dtype=torch.long)
        for i, sentence in enumerate(word_seq_list):
            for j, word in enumerate(sentence):
                for k, char in enumerate(word):
                    if char:
                        input_tensor[i, j, k] = ord(char)

        tokenized_samples = tokenizer.batch_encode_plus(
            all_samples_per_split["tokens"], is_split_into_words=True, max_length=256, padding='max_length', truncation=True, return_token_type_ids= False)
        total_adjusted_labels = []

        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split["ner_tags"][k]
            i = -1
            adjusted_label_ids = []

            for wid in word_ids_list:
                if(wid is None):
                    adjusted_label_ids.append(-100)
                elif(wid != prev_wid):
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = wid
                else:
                    adjusted_label_ids.append(existing_label_ids[i])

            total_adjusted_labels.append(adjusted_label_ids)

        tokenized_samples["labels"] = total_adjusted_labels
        tokenized_samples["charcnn"] = input_tensor
        tokenized_samples["rnn"] = input_tensor


        return tokenized_samples

    tokenizedTrainDataset = dataset["train"].map(
        tokenize_adjust_labels, batched=True)
    tokenizedTrainDataset = tokenizedTrainDataset.remove_columns(["ner_tags", "tokens"])
    
    tokenizedValDataset = dataset["validation"].map(
        tokenize_adjust_labels, batched=True)
    tokenizedValDataset = tokenizedValDataset.remove_columns(["ner_tags", "tokens"])
    
    tokenizedTestDataset = dataset["test"].map(
        tokenize_adjust_labels, batched=True)
    tokenizedTestDataset = tokenizedTestDataset.remove_columns(["ner_tags", "tokens"])
    proxies = {
        'http': 'http://172.17.0.1:10809',
        'https': 'http://172.17.0.1:10809',
    }

    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(
            predictions=true_predictions, references=true_labels)
        flattened_results = {
            "overall_precision": results["overall_precision"],
            "overall_recall": results["overall_recall"],
            "overall_f1": results["overall_f1"],
            "overall_accuracy": results["overall_accuracy"],
        }

        return flattened_results

    return tokenizedTrainDataset, tokenizedValDataset, tokenizedTestDataset, compute_metrics, label_names
from torch.utils.data import DataLoader
def train_and_evaluate(lr,
                       batchsize,
                       epochs,
                       tokenizer,
                       tokenizedTrainDataset,
                       tokenizedValDataset,
                       tokenizedTestDataset,
                       compute_metrics,
                       label_names,
                       logsPath=None,
                       trainingArgs=None,
                       model = None):
    
    data_collator = DataCollatorForTokenClassification(tokenizer)

    model.train()

    if trainingArgs == None:
        trainingArguments = ts.TrainingArguments(
            "output/",
            seed=42,
            logging_steps=250,
            save_steps=2500,
            num_train_epochs=epochs,
            learning_rate=lr,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=batchsize,
            per_device_eval_batch_size=batchsize,
            weight_decay=0.01,
        )
    else:
        trainingArguments = trainingArgs

    trainer = ts.Trainer(
        model=model,
        args=trainingArguments,
        train_dataset=tokenizedTrainDataset,
        eval_dataset=tokenizedValDataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.eval()

    evaluationResult = trainer.evaluate()

    trainer.eval_dataset = tokenizedTestDataset

    testResult = trainer.evaluate()

    if logsPath != None:
        with open(logsPath, mode="a+") as f:
            f.write(
                f"---HyperParams---\nBatchsize= {batchsize} Lr= {lr}\n---Val Results---\n{str(evaluationResult)}\n---Test Results---\n{str(testResult)}\n\n")

    return model, evaluationResult, testResult

