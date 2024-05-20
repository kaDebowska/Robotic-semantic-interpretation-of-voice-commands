import csv
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

def extract_commands_from_file(file_path):
    commands = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        lines = file.readlines()
        command = ""
        for line in lines:
            if line.strip().startswith("#"):
                continue
            elif line.strip() == "":
                continue
            else:
                command += line.strip() + " "
                if line.strip().endswith("."):
                    commands.append(command.strip())
                    command = ""
    return commands


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


file_path = "generated_commands.txt"
commands = extract_commands_from_file(file_path)
create_new_csv('train.csv', commands)

data = pd.read_csv('train.csv')

train_data = data.sample(frac=0.7, random_state=42)
data = data.drop(train_data.index)
val_data = data.sample(frac=0.5, random_state=42)
test_data = data.drop(val_data.index)

train_data.to_csv('train.csv', index=False)
val_data.to_csv('val.csv', index=False)
test_data.to_csv('test.csv', index=False)