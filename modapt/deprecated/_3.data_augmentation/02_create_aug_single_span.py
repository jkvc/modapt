from collections import defaultdict
from os import mkdir
from os.path import exists, join

import pandas as pd
from config import AUG_SINGLE_SPANS_DIR, FRAMING_DATA_DIR, ISSUES
from tqdm import tqdm

from modapt.utils import load_json, save_json

MIN_SPAN_NUM_CHAR = 30

if __name__ == "__main__":
    if not exists(AUG_SINGLE_SPANS_DIR):
        mkdir(AUG_SINGLE_SPANS_DIR)

    stats = {}

    for issue in ISSUES:
        print(">>", issue)
        data = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))
        codes = load_json(join(FRAMING_DATA_DIR, "codes.json"))

        labeled_span_data = defaultdict(list)

        for articleid, article in tqdm(data.items()):
            text = article["text"]
            annotations = article["annotations"]
            framing_annotations = annotations["framing"]

            for annotator, spans in framing_annotations.items():
                for span in spans:
                    start = span["start"]
                    end = span["end"]
                    if end - start < MIN_SPAN_NUM_CHAR:
                        continue
                    code = span["code"]
                    text_segment = text[start:end]
                    labeled_span_data[articleid].append(
                        {
                            "text": text_segment,
                            "code": code,
                        }
                    )
        save_json(
            labeled_span_data,
            join(
                AUG_SINGLE_SPANS_DIR,
                f"{issue}_frame_spans_min{MIN_SPAN_NUM_CHAR}.json",
            ),
        )

        num_spans = sum(len(l) for l in labeled_span_data.values())
        stats[issue] = {
            "num_spans": num_spans,
            "num_articles": len(labeled_span_data),
        }
        print(len(labeled_span_data), num_spans)

    df = pd.DataFrame.from_dict(stats, orient="index")
    df["ratio"] = df["num_spans"] / df["num_articles"]
    df.to_csv(
        join(
            AUG_SINGLE_SPANS_DIR,
            f"stats_spans_min{MIN_SPAN_NUM_CHAR}.csv",
        )
    )
