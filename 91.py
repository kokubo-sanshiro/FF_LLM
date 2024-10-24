import os
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import evaluate
from datasets import load_dataset, Dataset

from huggingface_hub import login
from transformers import pipeline

"""
AutoTokenizer: 指定したモデルに応じて自動的に適切なトークナイザーをロードしてくれる。
AutoModel: 指定したモデルに応じて自動的に適切なモデルをロードしてくれる。

https://huggingface.co/docs/transformers/tasks/translation

https://note.com/npaka/n/n17ecbd890cd6#9ZILg

from huggingface_hub import login
"""




def main():
    data_path = r"../FF_LLM_data/kftt-data-1.0/data/tok"

    ##'Helsinki-NLP/opus-mt-ja-en'
    model_weight = "Helsinki-NLP/opus-mt-ja-en"

    login(token="")

    prefix = "日本語 を 英語 に 翻訳 して ください 。: "


    # llm = LLM(data_path = data_path, model_weight = model_weight,  prefix=prefix)
    # llm.train(output_dir = "../logs/ja2en_helsinki")


    llm = LLM(data_path = data_path, model_weight = r"C:\Users\kokub\2024\logs\ja2en_helsinki", prefix=prefix)
    print(llm.inference("私 は 本 を 読み ます 。"))


class LLM():
    def __init__(self, data_path, model_weight =  "google-t5/t5-small",  source_lang = "ja", target_lang = "en", prefix = "日本語 を 英語 に 翻訳 して ください 。: "):
        self.model_weight = model_weight
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_weight)

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.prefix      = prefix

        self.train_dataset, self.test_dataset, self.dev_dataset = self.make_dataset(data_path)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_weight)
    

    def inference(self, text, max_new_tokens=40):

        # テキストをトークン化してテンソル形式に変換
        inputs = self.tokenizer(self.prefix + text, return_tensors="pt").input_ids

        # 生成
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  
        )

        # 生成結果をデコード
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated_text



    def make_data_path(self, data_path):
        # データファイルパスを生成
        train_ja_path = os.path.join(data_path, "kyoto-train.ja")
        train_en_path = os.path.join(data_path, "kyoto-train.en")
        test_ja_path = os.path.join(data_path, "kyoto-test.ja")
        test_en_path = os.path.join(data_path, "kyoto-test.en")
        dev_ja_path = os.path.join(data_path, "kyoto-dev.ja")
        dev_en_path = os.path.join(data_path, "kyoto-dev.en")

        return train_ja_path, train_en_path, test_ja_path, test_en_path, dev_ja_path, dev_en_path

    def make_dataset(self, data_path):
        # データパスを取得
        train_ja_path, train_en_path, test_ja_path, test_en_path, dev_ja_path, dev_en_path = self.make_data_path(data_path)

        # データセットを読み込み
        train_dataset = self.load_translation_dataset(train_ja_path, train_en_path)
        test_dataset = self.load_translation_dataset(test_ja_path, test_en_path)
        dev_dataset = self.load_translation_dataset(dev_ja_path, dev_en_path)

        return train_dataset, test_dataset, dev_dataset

    def load_translation_dataset(self, ja_file_path, en_file_path):
        # jaとenのデータを読み込む
        dataset = load_dataset(
            "text",
            data_files={"ja": [ja_file_path], "en": [en_file_path]},
        )

        # 翻訳データを整形
        translation_data = [
            {"translation": {"ja": ja["text"], "en": en["text"]}}
            for ja, en in zip(dataset["ja"], dataset["en"])
        ]

        # Datasetオブジェクトに変換
        translation_dataset = Dataset.from_list(translation_data)


        return translation_dataset

    def preprocess_function(self, examples):
        inputs = [self.prefix + example[self.source_lang] for example in examples["translation"]]
        targets = [example[self.target_lang] for example in examples["translation"]]
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs
    
    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels
    
    def compute_metrics(self, eval_preds):
        metric = evaluate.load("sacrebleu")

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    def train(self, output_dir):
        
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model_weight)

        train_tokenizer = self.train_dataset.map(self.preprocess_function, batched=True, remove_columns=["translation"])
        test_tokenizer = self.test_dataset.map(self.preprocess_function, batched=True, remove_columns=["translation"])
        
        training_args = Seq2SeqTrainingArguments(
            output_dir                  = output_dir,
            eval_strategy               ="epoch",
            learning_rate               = 2e-5,
            per_device_train_batch_size = 16,
            per_device_eval_batch_size  = 16,
            weight_decay                = 0.01,
            save_total_limit            = 3,
            num_train_epochs            = 10,
            predict_with_generate       = True,
            fp16                        = True, #change to bf16=True for XPU
            push_to_hub                 = True,
        )

        trainer = Seq2SeqTrainer(
            model           = self.model,
            args            = training_args,
            train_dataset   = train_tokenizer,
            eval_dataset    = test_tokenizer,
            tokenizer       = self.tokenizer,
            data_collator   = data_collator,
            compute_metrics = self.compute_metrics,
        )

        trainer.train()
        trainer.push_to_hub()
    

main()









