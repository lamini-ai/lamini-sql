from llama.error.error import APIError as LlamaAPIError

from llama import LLM, Type, Context

import json
import random
import jsonlines

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="Lamini-SQL", description="Generates SQL data for LLM instruction tuning"
    )

    parser.add_argument(
        "-c", "--count", default=100, help="The number of examples to generate."
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=10,
        help="The number of examples to generate in a batch.",
    )

    arguments = vars(parser.parse_args())

    total_examples = int(arguments["count"])
    batch_size = int(arguments["batch_size"])

    dataset = load_spider()

    save_spider()

    for count in range(0, total_examples, batch_size):
        if count + batch_size > total_examples:
            batch_size = total_examples - count
        print(
            f"Processing index {count} out of {total_examples} using batch size {batch_size}"
        )
        generate_questions(start_index=count, batch_size=batch_size, dataset=dataset)
        generate_queries(index=count, batch_size=batch_size)


def generate_questions(start_index, batch_size, dataset):
    with open("data/questions.jsonl", "a") as questions_file:
        writer = jsonlines.Writer(questions_file, flush=True)

        llm = LLM(name="generate-sql")

        llm.add_data(make_pairs(dataset))

        for index in range(start_index, start_index + batch_size):
            item = dataset[index % len(dataset)]

            print("====== Seed Question =====\n", item)
            novel_question = get_question(llm, item)
            novel_question.question = parse(novel_question.question)
            print("===== Novel Question =====\n", novel_question)
            writer.write(novel_question.dict())


def get_question(llm, item):

    attempts = 10

    for i in range(attempts):
        try:
            return llm(
                input=item,
                output_type=NovelQuestion,
                temperature=0.7,
                model_name="lamini/open",
                max_tokens=32,
            )
        except LlamaAPIError as e:
            print("Lamini API error {i}, retrying")

    raise RuntimeError("Too many Lamini API errors.")


def parse(string):
    # position = string.find("\n")

    # string = string[position + 1 :]

    position = string.find("\n", 10)
    if position > 0:
        string = string[:position]

    position = string.find(".", 10)
    if position > 0:
        string = string[:position]

    return string


def make_pairs(dataset):
    pairs = []
    for seed in dataset:
        other = random.sample(dataset, 1)[0]

        pairs.append([seed, NovelQuestion(question=other["question"])])

    return pairs


class Question(Type):
    question: str = Context("a question about a sql table")


class NovelQuestion(Type):
    question: str = Context("a novel question, about a different table")


def load_spider():
    items = []
    with open("data/spider/train_spider.json") as dataset_file:
        dataset = json.load(dataset_file)
        for item in dataset:
            items.append(Question(question=item["question"]))

    return items


def save_spider():
    with open("data/spider/train_spider.json") as spider_file:
        dataset = json.load(spider_file)
        with open("data/dataset.jsonl", "w") as dataset_file:
            writer = jsonlines.Writer(dataset_file, flush=True)
            for item in dataset:
                question_and_query = QuestionAndQuery(
                    question=item["question"], query=item["query"]
                )
                print("===== Reference Question and Query =====\n", question_and_query)
                writer.write(question_and_query.dict())


class Query(Type):
    query: str = Context("the sql query to answer the question")


class QuestionAndQuery(Type):
    question: str = Context("a question about a sql table")
    query: str = Context("the sql query to answer the question")


def generate_queries(index, batch_size):
    questions = list(load_questions(path="data/questions.jsonl"))

    with open("data/dataset.jsonl", "a") as dataset_file:
        writer = jsonlines.Writer(dataset_file, flush=True)

        llm = LLM(name="generate-lamini-sql-query")

        llm.add_data(load_query_data())

        for question in questions[index : index + batch_size]:
            print("====== Question =====\n", question)
            query = get_query(llm, question)

            query.query = parse_query(query.query)
            print("===== Response =====\n", query)
            question_and_query = QuestionAndQuery(
                question=question.question, query=query.query
            )
            writer.write(question_and_query.dict())


def load_questions(path, key="question"):
    with open(path) as questions_file:
        reader = jsonlines.Reader(questions_file)

        for index, line in enumerate(reader):
            yield Question(
                question=line[key],
            )


def load_query_data():
    items = []
    with open("data/spider/train_spider.json") as dataset_file:
        dataset = json.load(dataset_file)
        for item in dataset:
            items.append(
                [Question(question=item["question"]), Query(query=item["query"])]
            )

    return items


def get_query(llm, question):

    attempts = 10

    for i in range(attempts):
        try:
            return llm(
                input=question,
                output_type=Query,
                temperature=0.0,
                model_name="lamini/open",
                max_tokens=128,
            )
        except LlamaAPIError as e:
            print("Lamini API error {i}, retrying")

    raise RuntimeError("Too many Lamini API errors.")


def parse_query(string):
    break_point = string.find("\n")

    if break_point >= 0:
        string = string[:break_point]

    return string.strip()


main()
