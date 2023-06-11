# References
    # https://rajpurkar.github.io/SQuAD-explorer/
    # we represent the input question and passage as a single packed sequence, with the question using the A embedding and the passage using the B embedding. We only introduce a start vector S 2 RH and an end vector E 2 RH during fine-tuning. The probability of word i being the start of the answer span is computed as a dot product between Ti and S followed by a softmax over all of the words in the paragsraph: Pi = eS Ti P j eS Tj . The analogous formula is used for the end of the answer span. The score of a candidate span from position i to position j is defined as S Ti + E Tj , and the maximum scoring span where j   i is used as a prediction. The training objective is the sum of the log-likelihoods of the correct start and end positions. We fine-tune for 3 epochs with a learning rate of 5e-5 and a batch size of 32. Table 2 shows top leaderboard entries as well

# SQuAD 1.1, the previous version of the SQuAD dataset,
# contains 100,000+ question-answer pairs on 500+ articles.
# SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions
# written adversarially by crowdworkers to look similar to answerable ones.

import json
from fastapi.encoders import jsonable_encoder
# from pprint import pprint
from tqdm.auto import tqdm


def parse_squad(json_path):
    with open(json_path, mode="r") as f:
        raw_data = jsonable_encoder(json.load(f))

    data = list()
    for article in raw_data["data"]:
        for parags in article["paragraphs"]:
            ctx = parags["context"]
            for qa in parags["qas"]:
                que = qa["question"]
                ans = qa["answers"]
                if ans:
                    start_id = ans[0]["answer_start"]
                    end_id = start_id + len(ans[0]["text"])
                else:
                    start_id = end_id = 0
                data.append(
                    {"question": que, "context":ctx, "answer": {"start_index": start_id, "end_index": end_id}}
                )
    return data
json_path = "/Users/jongbeomkim/Downloads/train-v2.0.json"
data = parse_squad(json_path)