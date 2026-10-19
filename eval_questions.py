# Evaluation set: questions whose answers are known from the thesis data.
# type "number":   the answer must contain a number within `tolerance` of `expected`
# type "contains": the answer must contain the `expected` text
# type "judgment": a separate model call grades the answer against `rubric`

QUESTIONS = [
    # --- simple lookups ---
    {"id": "q01", "category": "lookup", "type": "number",
     "question": "How many recordings are in the dataset?", "expected": 32, "tolerance": 0},
    {"id": "q02", "category": "lookup", "type": "number",
     "question": "How many transitions are there in total?", "expected": 12910, "tolerance": 0},
    {"id": "q03", "category": "lookup", "type": "contains",
     "question": "Which recording has the fewest transitions?", "expected": "2025-04-21_20-02-27"},
    {"id": "q04", "category": "lookup", "type": "number",
     "question": "How many recordings were made on 2025-04-21?", "expected": 10, "tolerance": 0},
    {"id": "q05", "category": "lookup", "type": "number",
     "question": "How many recordings have 8 frogs?", "expected": 3, "tolerance": 0},

    # --- aggregations ---
    {"id": "q06", "category": "aggregation", "type": "number",
     "question": "What is the median response delay (time_difference_j_k) in milliseconds?",
     "expected": 20, "tolerance": 0},
    {"id": "q07", "category": "aggregation", "type": "number",
     "question": "What is the mean response delay in milliseconds on 2025-04-21?",
     "expected": 51.98, "tolerance": 0.5},
    {"id": "q08", "category": "aggregation", "type": "contains",
     "question": "Which recording has the loudest calls on average (peak_call_db_k)?",
     "expected": "test1_2023-03-03_18-44-18"},
    {"id": "q09", "category": "aggregation", "type": "number",
     "question": "What fraction of transitions return to the frog that called two calls earlier "
                 "(frog_i equals frog_k)? Give it as a percentage.", "expected": 42.0, "tolerance": 0.5},
    {"id": "q10", "category": "aggregation", "type": "number",
     "question": "What is the average number of frogs per recording?", "expected": 5.625, "tolerance": 0.05},

    # --- data quality ---
    {"id": "q11", "category": "data_quality", "type": "number",
     "question": "How many transitions have a negative response delay?", "expected": 111, "tolerance": 0},
    {"id": "q12", "category": "data_quality", "type": "number",
     "question": "In how many recordings does frog id 0 appear?", "expected": 5, "tolerance": 0},

    # --- traps: the right answer is a caveat, not a number ---
    {"id": "q13", "category": "trap", "type": "judgment",
     "question": "Are frogs farther apart on 2025-04-21 than on 2023-03-03?",
     "rubric": "Mean distance is about 0.45 on 2023-03-03 and about 12.7 on 2025-04-21 (~30x). "
               "A correct answer reports the gap AND warns that distances may not be comparable "
               "between the two nights (different scale or estimation method). "
               "Stating that frogs were simply farther apart, with no warning, is INCORRECT."},
    {"id": "q14", "category": "trap", "type": "judgment",
     "question": "What is the average peak frequency of frog 3?",
     "rubric": "Frog ids are assigned separately in each recording, so 'frog 3' is not one individual. "
               "A correct answer points this out (and may give per-recording values). "
               "A single average over all rows with frog id 3, presented as one frog, is INCORRECT."},
    {"id": "q15", "category": "trap", "type": "judgment",
     "question": "Do frogs respond faster to closer neighbors?",
     "rubric": "Within recordings the correlation between distance_j_k and time_difference_j_k is "
               "near zero (median Spearman about -0.02 across 32 recordings). A correct answer analyses "
               "it within recordings (or controls for recording/date) and concludes there is no clear "
               "effect. Pooling both nights and reporting a strong effect is INCORRECT."},
]