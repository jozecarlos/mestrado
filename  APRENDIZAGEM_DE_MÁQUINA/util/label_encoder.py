class Label:
    def __init__(self, label, encode):
        self.label = label
        self.encode = encode


class LabelEncoder:
    @staticmethod
    def categorical(data, classes):
        labels = {}
        result = []
        class_tmp = sorted(classes)
        for i in range(len(classes)):
            labels[class_tmp[i]] = i

        for j in range(len(data)):
            result.append(labels[data[j]])

        return result

    @staticmethod
    def one_hot_encoding():
        print("One Hot Encoding")
