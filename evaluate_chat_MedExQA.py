import os
import argparse
import re
import torch
import pandas as pd
from tqdm import tqdm
from thefuzz import process
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        pad_token_id=tokenizer.pad_token_id,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True
    )
    model.generation_config.do_sample = False  # use greedy decoding
    model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
    return model, tokenizer


def format_example(line):
    example = (
        "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question. Your answer should be paired with an explanation why you chose that answer.\n\n"
        + line["question"]
        + "\n"
    )
    for choice in choices:
        example += f'{choice}. {line[f"{choice}"]}\n'
    return example


def process_before_extraction(gen, choice_dict):
    # replace the choice by letter in the generated sentence
    # from longest one to shortest one
    for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
        pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
        gen = pattern.sub(key, gen)
    return gen


def extract_choice(gen, choice_list):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)

    # simply extract the first appearred letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)


def extract_answer(response, row):
    gen = process_before_extraction(
        response, {choice: row[choice] for choice in choices}
    )
    pred = extract_choice(gen, [row[choice] for choice in choices])
    return pred


@torch.no_grad()
def eval_subject(
    model,
    tokenizer,
    subject_name,
    test_df,
    save_result_dir=None,
    overwrite=False,
    **kwargs
):
    result_path = os.path.join(save_result_dir, f"{subject_name}_result.csv")
    if not overwrite and os.path.exists(result_path):
        print(f"{result_path} existed, skip!")
        score = []
        for (_, datarow), (_, resultrow) in zip(
            test_df.iterrows(), pd.read_csv(result_path).iterrows()
        ):
            # pred = extract_answer(resultrow['model_response'], datarow)
            pred = resultrow["model_output"]
            correct = 1 if pred == datarow["answer"] else 0
            score.append(correct)
        return score

    result = []
    score = []
    responses = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = format_example(row)
        inputs = tokenizer(question, return_tensors="pt", return_attention_mask=False).to(device)

        #여기를 바꿔야함
        outputs = model.generate(**inputs, 
                                 do_sample=False,
                                 temperature=0.0, 
                                 top_p=None,
                                 num_beams=1,
                                 no_repeat_ngram_size=3,
                                 eos_token_id=tokenizer.eos_token_id,  # End of sequence token
                                 pad_token_id=tokenizer.eos_token_id,  # Pad token
                                 max_new_tokens=300,
                                )               
        
        response = tokenizer.batch_decode(outputs)[0]
        pred = extract_answer(response, row)
        responses.append(response)

        if "answer" in row:
            correct = 1 if pred == row["answer"] else 0
            score.append(correct)
            if args.debug:
                print(f'{question} pred: {pred} ref: {row["answer"]}')
        result.append(pred)

    if save_result_dir:
        test_df["model_output"] = result
        test_df["model_response"] = responses
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(
            os.path.join(save_result_dir, f"{subject_name}_result.csv"),
            encoding="utf-8",
            index=False,
        )

    return score

def main(args):
    print("loading model weights")
    if args.checkpoint_path is not None:
        model, tokenizer = load_models_tokenizer(args)
    else:
        model, tokenizer = None, None
    print("model loaded")

    dev_result = {}
    for subject_name in tqdm(SUBJECTS):
        if os.path.isfile(os.path.join(args.output_result_dir, f"{subject_name}_result.csv")):
            df = pd.read_csv(os.path.join(args.output_result_dir, f"{subject_name}_result.csv"))
            score = df["correctness"].tolist()
            dev_result[subject_name]=score
            continue
        
        test_file_path = os.path.join(
            args.eval_data_path, "test", f"{subject_name}_test.tsv"
        )
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "exp0", "exp1", "answer"], sep='\t'
        )

        score = eval_subject(
            model,
            tokenizer,
            subject_name,
            test_df,
            save_result_dir=args.output_result_dir,
            overwrite=args.overwrite,
        )
        dev_result[subject_name] = score


SUBJECTS = [
    "biomedical_engineer",
    "clinical_psychologist",
    "speech_pathologist",
    "occupational_therapist",
    "clinical_laboratory_scientist",
]
choices = ["A", "B", "C", "D"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")

    # Provide extra arguments required for tasks
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval_data_path", type=str, help="Path to eval data")
    group.add_argument(
        "--debug", action="store_true", default=False, help="Print infos."
    )
    group.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existed results",
    )
    group.add_argument("-o", "--output_result_dir", type=str, help="Path to output result dir")

    args = parser.parse_args()
    set_seed(args.seed)

    main(args)
