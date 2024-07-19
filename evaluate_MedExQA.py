import os
from typing import List
import pandas as pd
import numpy as np
import argparse
import torch
from tqdm import tqdm
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

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
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True
    )
    return model, tokenizer


def format_example(line, include_answer=True):
    example = "Question: " + line["question"]
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += "\nAnswer: " + line["answer"] + "\n\n"
    else:
        example += "\nAnswer:"
    return example


def generate_few_shot_prompt(k, subject, dev_df):
    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s.strip()

    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )

    if k == -1:
        k = dev_df.shape[0]
    for i in range(k):
        prompt += format_example(
            dev_df.iloc[i, :],
            include_answer=True,
        )
    return prompt


def get_logits(tokenizer, model, inputs: List[str]):
    input_ids = tokenizer(inputs, padding='longest')["input_ids"]
    input_ids = torch.tensor(input_ids, device=model.device)

    if input_ids.shape[1] > args.max_seq_len:
        input_ids = input_ids[:, input_ids.shape[1] - args.max_seq_len + 1 :]
    tokens = {"input_ids": input_ids}
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    outputs = model(input_ids, attention_mask=attention_mask)["logits"]
    logits = outputs[:, -1, :]
    log_probs = torch.nn.functional.softmax(logits, dim=-1)
    return log_probs, {"tokens": tokens}


@torch.no_grad()
def eval_subject(
    model,
    tokenizer,
    subject_name,
    test_df,
    k=5,
    dev_df=None,
    few_shot=False,
    save_result_dir=None,
    batch_size=1,
    **kwargs,
):
    result = []
    score = []

    few_shot_prompt = (
        generate_few_shot_prompt(k, subject_name, dev_df) if few_shot else []
    )
    all_probs = {"prob_A": [], "prob_B": [], "prob_C": [], "prob_D": []}
    if args.debug:
        print(f"few_shot_prompt: {few_shot_prompt}")

    choices_ids = torch.tensor(
        tokenizer(["A", "B", "C", "D"], add_special_tokens=False)["input_ids"]).flatten().unsqueeze(0).to(model.device)
    idx_list = list(range(0, len(test_df), batch_size))
    for i in tqdm(idx_list):
        full_prompt_list = []
        answer_list = []
        for row in test_df.iloc[i:i+batch_size].to_dict(orient='records'):
            question = format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + question
            full_prompt_list.append(full_prompt)
            if 'answer' in row:
                answer_list.append(row['answer'])

        logits, input_info = get_logits(tokenizer, model, full_prompt_list)
        softval = logits.gather(1, choices_ids.expand(logits.size(0), -1)).softmax(1)
        if softval.dtype in {torch.bfloat16, torch.float16}:
            softval = softval.to(dtype=torch.float32)
        probs = softval.detach().cpu().numpy()
        for i in range(len(probs)):
            for j, choice in enumerate(choices):
                all_probs[f"prob_{choice}"].append(probs[i][j])
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs[i])]

            if answer_list != []:
                correct = 1 if pred == answer_list[i] else 0
                score.append(correct)
                if args.debug:
                    print(f'{question} pred: {pred} ref: {answer_list[i]}')
            result.append(pred)

    if save_result_dir:
        test_df["model_output"] = result
        for i, choice in enumerate(choices):
            test_df[f"prob_{choice}"] = all_probs[f"prob_{choice}"]
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
    model, tokenizer = load_models_tokenizer(args)

    dev_result = {}
    for subject_name in tqdm(SUBJECTS):
        if os.path.isfile(os.path.join(args.output_result_dir, f"{subject_name}_result.csv")):
            df = pd.read_csv(os.path.join(args.output_result_dir, f"{subject_name}_result.csv"))
            score = df["correctness"].tolist()
            dev_result[subject_name]=score
            continue
            
        dev_file_path = os.path.join(
            args.eval_data_path, "dev", f"{subject_name}_dev.tsv"
        )
        test_file_path = os.path.join(
            args.eval_data_path, "test", f"{subject_name}_test.tsv"
        )
        
        dev_df = pd.read_csv(
            dev_file_path, names=["question", "A", "B", "C", "D", "exp0", "exp1", "answer"], sep='\t'
        )
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "exp0", "exp1", "answer"], sep='\t'
        )
        
        score = eval_subject(
            model,
            tokenizer,
            subject_name,
            test_df,
            dev_df=dev_df,
            k=0,#5,
            few_shot=True,
            save_result_dir=args.output_result_dir,
            batch_size=args.batch_size
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
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")

    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval_data_path", type=str, help="Path to eval data")
    group.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--debug", action="store_true", default=False, help="Print infos."
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batch size",
    )
    group.add_argument("-o", "--output_result_dir", type=str, help="Path to output result dir")

    args = parser.parse_args()
    set_seed(args.seed)

    main(args)
