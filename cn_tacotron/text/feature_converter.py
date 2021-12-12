from cn_tacotron.text.phones_mix import phone_to_id


def label_to_sequence(label):
    phones = []

    with open(label, 'r', encoding='utf8') as lab_reader:
        line = lab_reader.readline()
        meta = line.strip().split('|')[2].split(' ')
        for item in meta:
            if item not in phone_to_id.keys():
                continue
            phones.append(item)

    phones.append("~")

    return [phone_to_id[p] for p in phones]


if __name__ == '__main__':
    result = label_to_sequence("../datasets/labels/000001.lab")
    print(result)