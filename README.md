# [Assignment 7: Text classification](https://snlp2019.github.io/a7/)

__Deadline:__ Aug 21, 10:00 CEST.

This task is about text classification.
In particular, given a short text,
we are interested in predicting the gender of the author.

Unlike the earlier assignments,
this assignment is intentionally open-ended.
You can use any of the methods we studied in the class,
as well as any data and resources outside the data provided here.

The main challenge in this assignment is the size of the data set.
Since our data set is small,
achieving reliably good scores is rather difficult.

## Data

The primary data set we will use is from the short essays 
written by the class participants at the beginning of the semester.
You will find two files in your repositories,
[one for training](essays-train.tsv)
and [one for testing](essays-test.tsv).
Both data sets are provided as tab-separated-value files.
Where column headers for the class label and the essay texts are
`label` and `text`, respectively.
Note that a record may span multiple lines,
if the essay text contains newlines,
in which case text will be quoted.
You are recommended to use a library (e.g., python `csv` library)
to read the data.

You should use the `essays-train.tsv` for both training and tuning your model,
and you should use `essays-test.tsv` **only for testing**.
It is OK to use it to test alternative models/systems,
but test file should not be used during development/tuning process.

## Task

### 7.1 Implement, tune and test a text classification model

Implement one or more text classification methods,
tune its hyperparameters,
and test on the provided test set.

Since the data set is small,
you are strongly recommended to find ways to make use of external data.
Potentially useful external data include, but not limited to,
earlier data sets on author profiling or gender detection,
pre-trained embeddings or features based on clustering results on a large, unlabeled data set.

You are encouraged to try multiple models/systems,
and organize your code the way you like.
Please pay attention to good programming practices.
You may lose points if your code is difficult to follow.

### 7.2 Report your results

Part of your grade will be based on a short (no more than 1000 words) report
describing your approach and your results.
Please write your report as a [markdown](https://en.wikipedia.org/wiki/Markdown)
document with name `report.md`.

Your report should include the following.

- Technical information about your code:
    - How to run your code to replicate/reproduce the results you report
    - A list of additional libraries required.
        If you use non-standard libraries, including them 
        in your repository is a good idea.
    - Links to additional data or resources used.
        If they are publicly available (and large),
        do not include them in your repository but provide links.
        If they are not publicly available,
        but too big for inclusion in your repository,
        please provide an alternative download link.
- A description of your approach:
    - Which method/model(s) did you use?
    - How did you tune your model(s)?
    - Did you employ any preprocessing or feature selection steps?
    - Did you use external data, if so what are they?
- A report of your results:
    - Report the appropriate scores for your model(s)
    - If you used multiple models,
        or alternative data sources and/or other settings
        with the same model, please report all.
    - A comparison with an appropriate trivial baseline.
- (optional) an analysis of useful features.
    For example by analyzing the learned weights,
    or through ablation experiments.
    
## Evaluation

- Half of the score will be based on checking your source code and experimental setup,
    and reproducibility of your results.
- You will get 2 points (out of 10) for a well-written report.
    The report should clearly indicate the required information above,
    and you should pay attention to formatting and the language.
- 3 points (out of 10) will be based on the macro-averaged F1 score on the test set.
    - In case of multiple results reported,
        only the best score from each team will be considered.
    - If the best score you report cannot be reproduced,
        the score will be set to the score of the random baseline.
        Your result will be considered "reproduced" if                                                                                         
        the observed score is within 5% of the reported one.
        Some variation expected, particularly in this small data set
        due to, e.g., differences in initial weights.
    - You will get a minimum of 1 point if your score is better
        than the random baseline.
    - You will get 2 points if your score is within ±1 standard deviation 
        from the mean score of all submissions.
    - You will get 3 points if your score is more than one standard
        deviation above the scores of all submissions.
