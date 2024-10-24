import os

data_path = r"../FF_LLM_data/kftt-data-1.0/data/tok"


def load_data(ja_path, en_path):
    with open(ja_path, 'r', encoding='utf-8') as f:
        ja_lines = [line.strip() for line in f.readlines()] 

    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f.readlines()]
    return ja_lines, en_lines

kyoto_dev_ja_path = os.path.join(data_path, "kyoto-dev.ja")
kyoto_dev_en_path = os.path.join(data_path, "kyoto-dev.en")

kyoto_test_ja_path = os.path.join(data_path, "kyoto-test.ja")
kyoto_test_en_path = os.path.join(data_path, "kyoto-test.en")

kyoto_train_ja_path = os.path.join(data_path, "kyoto-train.ja")
kyoto_train_en_path = os.path.join(data_path, "kyoto-train.en")


ja_train, en_train = load_data(kyoto_train_ja_path, kyoto_train_en_path)
ja_dev, en_dev = load_data(kyoto_dev_ja_path, kyoto_dev_en_path)

print(ja_train[0])

