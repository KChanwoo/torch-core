import random


class DatasetDivider:
    def __init__(self, class_idx=None, train_test_ratio=.2, train_val_ratio=.1, label_key_name="label",
                 data_key_name="data"):
        self.train_test_ratio = train_test_ratio
        self.train_val_ratio = train_val_ratio
        self.class_idx = class_idx
        self.label_key_name = label_key_name
        self.data_key_name = data_key_name

    def __call__(self, data, labels, shuffle=True):
        train_data = []
        val_data = []
        test_data = []

        train_val_len = int(len(data) * self.train_test_ratio)
        train_len = int(train_val_len) * self.train_val_ratio

        if self.class_idx:
            for key in self.class_idx.keys():
                class_data = [{self.data_key_name: data[i], self.label_key_name: labels[i]} for i in range(len(labels)) if labels[i] == key]

                if shuffle:
                    random.shuffle(class_data)

                train_data.extend(class_data[:train_len])
                val_data.extend(class_data[train_len: train_val_len])
                test_data.extend(class_data[train_val_len: len(class_data)])
        else:
            class_data = [{self.data_key_name: data[i], self.label_key_name: labels[i]} for i in range(len(labels))]
            if shuffle:
                random.shuffle(data)

            train_data.extend(class_data[:train_len])
            val_data.extend(class_data[train_len: train_val_len])
            test_data.extend(class_data[train_val_len: len(class_data)])

        return train_data, val_data, test_data
