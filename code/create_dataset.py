import pandas as pd
import json
from sklearn.model_selection import train_test_split

projectquestion_data = pd.read_excel(
    "finetune-llms/data/projectquestion_examples.xlsx"
)

def create_jsonl_file(data: pd.DataFrame, file_name: str, sep: str) -> None:
    """
    Function to create the jsonl training and testing files
    """
    with open(
        file_name,
        "w",
        encoding="utf-8",
    ) as f:
        for index, row in data.iterrows():
            json.dump(
                {
                    "input_text": prompt + row["input_text"] + prompt_end,
                    "output_text": f"\n## {sep} ##\n"
                    + row["output_text"]
                    + "\n\n </s>",
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

# depends for which model you want to create the dataset
model = "LeoLM"

if model == "Llama2":
    prompt = """"<s>[INST] <<SYS>>
        You are a helpful assistant for toom Baumarkt, a DIY-store in Germany.
        Users will give you a description of their planned projects delimited by a ## DESCRIPTION ## delimiter and you try to create a list of needed tools
        as well as needed materials. Please format your output like this that is delimited like so:
        ## OUTPUT ##
        Werkzeuge:
        * <Werkzeug 1> | <Kurzbeschreibung Werkzeug 1>
        * <Werkzeug 2> | <Kurzbeschreibung Werkzeug 2>
        ...

        Materialien:
        * <Material 1> | <Kurzbeschreibung Material 1>
        * <Material 2> | <Kurzbeschreibung Material 2>
        ...

        Answer in German. Don't give any more info than the list of tools and materials
        <</SYS>> 
    ## DESCRIPTION ## 
    """
    prompt_end = " [/INST]"

    # split in training and testing
    train, test = train_test_split(projectquestion_data, test_size=0.1, random_state=3)

    create_jsonl_file(
        train,
        file_name="finetune-llms/data/projectquestion_Llama2_training.jsonl",
        sep="OUTPUT",
    )
    create_jsonl_file(
        test,
        file_name="finetune-llms/data/projectquestion_Llama2_testing.jsonl",
        sep="OUTPUT",
    )

else:
    prompt = """ " <|im_start|>system
    Du bist ein assistent für toom Baumarkt.
    Kunden geben dir eine Beschreibung ihrer Projekte getrennt durch ## BESCHREIBUNG ## und du antwortest mit einer Liste an benötigten Materialien und Werkzeugen.
    Bitte formatiere dein Antwort so:
    ## ANTWORT ##
    Werkzeuge:
    * <Werkzeug 1> | <Kurzbeschreibung Werkzeug 1>
    * <Werkzeug 2> | <Kurzbeschreibung Werkzeug 2>
        ...

    Materialien: 
    * <Material 1> | <Kurzbeschreibung Material 1>
    * <Material 2> | <Kurzbeschreibung Material 2>
    ...

    <|im_end|>
    <|im_start|> user ## BESCHREIBUNG ## "
    """

    prompt_end = " <|im_end|>"

    # split in training and testing
    train, test = train_test_split(projectquestion_data, test_size=0.1, random_state=3)

    create_jsonl_file(
        train,
        file_name="finetune-llms/data/projectquestion_LeoLM_training.jsonl",
        sep="ANTWORT",
    )
    create_jsonl_file(
        test,
        file_name="finetune-llms/data/projectquestion_LeoLM_testing.jsonl",
        sep="ANTWORT",
    )


