import numpy as np
from PIL import Image


def open_txt(path):
    lines = []
    file = open(path)
    raw = file.readlines()
    file.close()
    for line in raw:
        lines.append(line.split(' '))

    return lines


if __name__ == "__main__":
    data = open_txt('./extraction/ocr_car_numbers_rotulado.txt')
    image = []
    images = []
    cc = 0
    im = 0
    row = []
    for line in data:
        for el in line:
            im = im + 1
            if im < 1226:
                row.append(255 if el == '1' else 0)
                cc = cc + 1
                if cc == 35:
                    image.append(row)
                    cc = 0
                    row = []
                if len(image) == 35:
                    #images.append(image)
                    a = np.squeeze(np.asarray(image), axis=2)
                    im = Image.fromarray(a)
                    im.save('/images/ocr_car/' + line[len(line) - 1] + ".jpeg")
                    cc = 0
                    im = 0
                    row = []
                    image = []


