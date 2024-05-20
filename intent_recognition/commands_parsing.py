import csv
import re
import spacy

nlp = spacy.load("en_core_web_sm")


def filter_commands(file_path):
    regex = re.compile(r'\b(put|bring)\b', flags=re.IGNORECASE)
    selected_commands = []
    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        rows = list(csv_reader)
        for i in range(0, len(rows), 2):
            command = rows[i][0]
            label = rows[i + 1][0]
            if regex.search(label):
                selected_commands.append(command)

    return selected_commands


def extract_subject_and_target(command):
    doc = nlp(command)
    subject = ''
    target = ''
    for token in doc:
        if token.dep_ == "dobj" and subject == '':
            subject = token.text
        elif token.dep_ == "pobj" or (token.dep_ == "dative" and token.text != 'to'):
            target = token.text
    return subject, target


def create_new_csv(file_path, commands):
    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Command", "Subject", "Target"])
        for command in commands:
            subject, target = extract_subject_and_target(command)
            csv_writer.writerow([command, subject, target])


def main():
    train_commands = filter_commands('datasets/train.csv')
    test_commands = filter_commands('datasets/test.csv')
    val_commands = filter_commands('datasets/val.csv')

    create_new_csv('datasets/train_filtered.csv', train_commands)
    create_new_csv('datasets/test_filtered.csv', test_commands)
    create_new_csv('datasets/val_filtered.csv', val_commands)

    for command in train_commands:
        subject, target = extract_subject_and_target(command)
        print(f"Command: {command}")
        print(f"Subject: {subject}")
        print(f"Target: {target}")

    # for command in commands:
    #     doc = nlp(command)
    #     print(f"Command: {command}")
    #     print("Dependencies:", [token.dep_ for token in doc])


if __name__ == '__main__':
    main()
