import json
import os.path
from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import SlakeDataset, SlakeCorruptionDataset
from llava.constants import DEFAULT_IMAGE_TOKEN


class SlakeDatamodule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, df, batch_size: int = 32, num_workers: int = 0, corruption=False, corruption_probabilities: dict = {'blur':0,'brightness': 0,'noise': 0}, corruption_strength: dict = {'blur':'medium','brightness': 'medium','noise': 'medium'}):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.df = df
        self.corruption = corruption
        self.corruption_probabilities = corruption_probabilities
        self.corruption_strength = corruption_strength

    def setup(self, stage: Optional[str] = None):
        if self.corruption:
            self.train_dataset = SlakeCorruptionDataset(self.data_dir, self.df, corruption_probabilities=self.corruption_probabilities, corruption_strength=self.corruption_strength)
            self.val_dataset = SlakeCorruptionDataset(self.data_dir, self.df, corruption_probabilities=self.corruption_probabilities, corruption_strength=self.corruption_strength)
            self.test_dataset = SlakeCorruptionDataset(self.data_dir, self.df, corruption_probabilities=self.corruption_probabilities, corruption_strength=self.corruption_strength)
        else:
            self.train_dataset = SlakeDataset(self.data_dir, self.df)
            self.val_dataset = SlakeDataset(self.data_dir, self.df)
            self.test_dataset = SlakeDataset(self.data_dir, self.df)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


def get_slake_df(data_dir,test_folder_name, train_folder_name, 
                 val_folder_name, mod, split, split_category=None, split_value=None):
    # split_value = split_value.capitalize() if split_category == "content_type" else split_value
    # mod -> train, val, test
    dataset_name = {
        "train" : train_folder_name,
        "val" : val_folder_name,
        "test" : test_folder_name,
    }

    df = pd.read_json(data_dir / f"{dataset_name[mod]}")
    df = df.loc[df['q_lang'] == "en"]

    if split == "all":
        return df

    if mod == "train" or mod == "val" or (mod == "test" and split == "iid"):
        df = df.loc[df[split_category] != split_value]
    elif mod == "test" and split == "ood":
        df_test = df.loc[df[split_category] == split_value]
        # If the split value changes within one patient, we only filter the test set,
        # since otherwise we might have the same patient / image within training and test set
        if split_category in ["answer_type", "content_type"]:
            df = df_test
        # If the split values stays constant within one patient, we can also take the training set
        # into the ood test set since this does not imply having the same patient in training / test set
        # TODO: should we really do it like this?
        else:
            df_train = pd.read_json(data_dir / train_folder_name)
            df_train = df_train.loc[df_train['q_lang'] == "en"]
            df_train = df_train.loc[df_train[split_category] == split_value]
            df = pd.concat([df_test, df_train])
    return df


def get_ovqa_df(data_dir,test_folder_name, train_folder_name,
                 val_folder_name, mod, split, split_category=None, split_value=None):
    
    # mod -> train, val, test
    dataset_name = {
        "train" : train_folder_name,
        "val" : val_folder_name,
        "test" : test_folder_name,
    }

    # if split_category == "question_type":
    #     split_value_divided = split_value.split("_")
    #     split_value = split_value_divided[0].capitalize()
    #     # if the split category contains words comnbined with "_"
    #     # separate the words and capitalize them 
    #     # TODO: check if there is a better way to do this
    #     for index in range(1,len(split_value_divided)):
    #         split_value += " " + split_value_divided[index].capitalize()
    # elif split_category == "image_organ":
    #     # TODO: Check if we need to split the words as above
    #     split_value = split_value.upper()

    df = pd.read_json(data_dir / f"{dataset_name[mod]}")

    if split == "all":
        return df

    if mod == "train" or mod == "val" or (mod == "test" and split == "iid"):
        df = df.loc[df[split_category] != split_value]
    elif mod == "test" and split == "ood":
        df_test = df.loc[df[split_category] == split_value]
        # If the split value changes within one patient, we only filter the test set,
        # since otherwise we might have the same patient / image within training and test set
        if split_category in ["question_type"]: # there are no duplicates in "image_organ"
            df = df_test
        # If the split values stays constant within one patient, we can also take the training set
        # into the ood test set since this does not imply having the same patient in training / test set
        # TODO: should we really do it like this?
        else:
            df_train = pd.read_json(data_dir / train_folder_name)
            df_train = df_train.loc[df_train[split_category] == split_value]
            df = pd.concat([df_test, df_train])
    return df


def get_mimic_df(data_dir, test_folder_name, train_folder_name,
                val_folder_name, mod, split, split_category=None, split_value=None):
    # mod -> train, val, test
    dataset_name = {
        "train": train_folder_name,
        "val": val_folder_name,
        "test": test_folder_name,
    }

    df = pd.read_json(data_dir / f"{dataset_name[mod]}")

    if split == "all":
        return df

    if split == "sample":
        if mod == "train":
            return df.sample(n=min(20000, len(df)), random_state=123)
        else:
            return df.sample(n=min(5000, len(df)), random_state=123)

    if split == "sample_iid":
        if split_category == "age" and split_value == "young":
            df = df.loc[df[split_category] >= 60]
        if split_category == "age" and split_value == "old":
            df = df.loc[df[split_category] < 40]
        elif split_category == "ethnicity" and split_value == "nonwhite":
            df = df.loc[df[split_category] == "WHITE"]
        elif split_category == "ethnicity" and split_value == "nonwhite":
            df = df.loc[df[split_category] == "WHITE"]
        elif split_category == "ethnicity" and split_value == "white":
            df = df.loc[df[split_category] != "WHITE"]
            df = df.loc[df[split_category] != "UNKNOWN/OTHER"]
        else:
            df = df.loc[df[split_category] != split_value]

        if mod == "train":
            return df.sample(n=min(20000, len(df)), random_state=123)
        else:
            return df.sample(n=min(5000, len(df)), random_state=123)
    

    if split == "sample_ood":
        if split_category == "age" and split_value == "young":
            df = df.loc[df[split_category] < 40]
        if split_category == "age" and split_value == "old":
            df = df.loc[df[split_category] >= 60]
        elif split_category == "ethnicity" and split_value == "nonwhite":
            df = df.loc[df[split_category] != "WHITE"]
            df = df.loc[df[split_category] != "UNKNOWN/OTHER"]
        elif split_category == "ethnicity" and split_value == "white":
            df = df.loc[df[split_category] == "WHITE"]
        else:
            df = df.loc[df[split_category] == split_value]

        if mod == "train":
            return df.sample(n=min(20000, len(df)), random_state=123)
        else:
            return df.sample(n=min(5000, len(df)), random_state=123)
        
    if mod == "train" or mod == "val" or (mod == "test" and split == "iid"):
        if split_category == "age" and split_value == "young":
            df = df.loc[df[split_category] >= 60]
        elif split_category == "ethnicity" and split_value == "nonwhite":
            df = df.loc[df[split_category] == "WHITE"]
        else:
            df = df.loc[df[split_category] != split_value]
    elif mod == "test" and split == "ood":
        if split_category == "age" and split_value == "young":
            df = df.loc[df[split_category] < 40]
        elif split_category == "ethnicity" and split_value == "nonwhite":
            df = df.loc[df[split_category] != "WHITE"]
            df = df.loc[df[split_category] != "UNKNOWN/OTHER"]
        else:
            df = df.loc[df[split_category] == split_value]
    if split_category == "gender":
        df = df.loc[df["content_type"] != "gender"]
    return df


def get_lidc_df(data_dir,test_folder_name, train_folder_name,
                 val_folder_name, mod, split, split_category=None, split_value=None):
    # split_value = split_value.capitalize() if split_category == "content_type" else split_value
    # mod -> train, val, test
    dataset_name = {
        "train" : train_folder_name,
        "val" : val_folder_name,
        "test" : test_folder_name,
    }

    df = pd.read_json(data_dir / f"{dataset_name[mod]}", dtype=str)

    if split == "all":
        return df

    if split_category in ["manufacturer"]:
        if mod == "train" or mod == "val" or (mod == "test" and split == "iid"):
            df = df.loc[df[split_category] != split_value]
        elif mod == "test" and split == "ood":
            # Here we do not want to add anything from training to not mess up patients in training and test set
            df = df.loc[df[split_category] == split_value]
    elif split_category in ["texture"]:
        if split_value == "non-solid":
            split_values_numerical = ["1.0", "2.0"]
        else:
            raise ValueError(f"Invalid split value: {split_value}")
        if mod == "train" or mod == "val" or (mod == "test" and split == "iid"):
            df = df.loc[~df[f"{split_category}_majority"].isin(split_values_numerical)]
            df = df.loc[df["content_type"] != split_category]
        elif mod == "test" and split == "ood":
            # Here we do not want to add anything from training to not mess up patients in training and test set
            df = df.loc[df[f"{split_category}_majority"].isin(split_values_numerical)]
            df = df.loc[df["content_type"] != split_category]
    else:
        raise ValueError(f"Invalid split category: {split_category}")
    return df

def get_datamodule(data_dir:Path, ood_value:str,
                   test_folder_name:str,train_folder_name:str,
                   val_folder_name:str,mod:str, dataset_name:str, 
                   split:str, data_shift:str, batch_size:int, num_workers:int = 0, no_image=False, 
                   corruption=False, corruption_probabilities: dict = {'blur':0,'brightness': 0,'noise': 0}, 
                   corruption_strength: dict = {'blur':'medium','brightness': 'medium','noise': 'medium'}):
    
    json_file_path, json_file_name = get_json_filename(data_dir=data_dir,
                                  ood_value=ood_value,test_folder_name=test_folder_name, 
                                  train_folder_name=train_folder_name,val_folder_name=val_folder_name,
                                  dataset_name=dataset_name, mod = mod, split=split, data_shift=data_shift, no_image=no_image)
    print(json_file_path)
    df = pd.read_json(json_file_path)
    # TODO: probably only one datamodule for all datasets without check for dataset_name
    if dataset_name == "SLAKE":
        return SlakeDatamodule(data_dir=data_dir, batch_size=batch_size, df=df, num_workers=num_workers, corruption=corruption,corruption_probabilities=corruption_probabilities,corruption_strength=corruption_strength), json_file_name
    elif dataset_name == "OVQA":
        # TODO : rename this as datamodule
        return SlakeDatamodule(data_dir=data_dir, batch_size=batch_size, df=df, num_workers=num_workers, corruption=corruption,corruption_probabilities=corruption_probabilities,corruption_strength=corruption_strength), json_file_name
    elif dataset_name == "LIDC":
        # TODO : rename this as datamodule
        return SlakeDatamodule(data_dir=data_dir, batch_size=batch_size, df=df, num_workers=num_workers, corruption=corruption,corruption_probabilities=corruption_probabilities,corruption_strength=corruption_strength), json_file_name
    elif dataset_name == "MIMIC":
        # TODO : rename this as datamodule
        return SlakeDatamodule(data_dir=data_dir, batch_size=batch_size, df=df, num_workers=num_workers, corruption=corruption,corruption_probabilities=corruption_probabilities,corruption_strength=corruption_strength), json_file_name
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")


def convert_raw_to_final(df, save_path, no_image=False):
    final_data = []

    # Process each entry as a separate conversation
    for _, row in df.iterrows():
        qid = row["qid"]
        new_entry = {
            "id": str(qid),
            "image": f"imgs/" + row["img_name"],
            "conversations": [
                {
                    "from": "human",
                    "value": f"{DEFAULT_IMAGE_TOKEN}\n{row['question']}" if not no_image else row['question']
                },
                {
                    "from": "gpt",
                    "value": row["answer"]
                }
            ],
            "img_id": row["img_id"],
            "language": row["q_lang"],
            "location": row["location"],
            "modality": row["modality"],
            "answer_type": row["answer_type"],
            "base_type": row["base_type"],
            "content_type": row["content_type"],
        }
        final_data.append(new_entry)

    with open(str(save_path), 'w') as output_file:
        json.dump(final_data, output_file, indent=4)


def convert_ovqa_raw_to_final(df, save_path, no_image=False):
    final_data = []

    # Process each entry as a separate conversation
    for _, row in df.iterrows():
        qid = row["qid"]
        new_entry = {
            "id": str(qid),
            "image": f"img/" + row["image_name"],
            "conversations": [
                {
                    "from": "human",
                    "value": f"{DEFAULT_IMAGE_TOKEN}\n{row['question']}" if not no_image else row['question']
                },
                {
                    "from": "gpt",
                    "value": row["answer"]
                }
            ],
            "img_id": row["image_name"].split(".")[0],
            "location": row["image_organ"],
            # TODO: we somehow need to add modality
            #"modality": row["modality"],
            "answer_type": row["answer_type"],
            "content_type": row["question_type"],
        }
        final_data.append(new_entry)

    with open(str(save_path), 'w') as output_file:
        json.dump(final_data, output_file, indent=4)


def convert_lidc_raw_to_final(df, save_path, no_image=False):
    final_data = []

    # Process each entry as a separate conversation
    for _, row in df.iterrows():
        qid = row["qid"]
        new_entry = {
            "id": str(qid),
            "image": f"images/" + f"{row['image_file_name']}.png",
            "conversations": [
                {
                    "from": "human",
                    "value": f"{DEFAULT_IMAGE_TOKEN}\n{row['question']}" if not no_image else row['question']
                },
                {
                    "from": "gpt",
                    "value": row["answer"]
                }
            ],
            "img_id": row["image_file_name"],
            "dicom_series_uid": row["dicom_series_uid"],
            "patient_id": row["patient_id"],
            "scan_id": row["scan_id"],
            "nodule_index": row["nodule_index"],
            "manufacturer": row["manufacturer"],
            "answer_type": row["answer_type"],
            "content_type": row["content_type"],
        }
        final_data.append(new_entry)

    with open(str(save_path), 'w') as output_file:
        json.dump(final_data, output_file, indent=4)


def convert_mimic_raw_to_final(df, save_path, no_image=False):
    final_data = []

    # Process each entry as a separate conversation
    for _, row in df.iterrows():
        qid = row["qid"]
        new_entry = {
            "id": str(qid),
            "image": f"mimic-cxr-vqa/physionet.org/files/mimic-cxr-jpg/2.0.0/files/" + f"{row['image_path']}",
            "conversations": [
                {
                    "from": "human",
                    "value": f"{DEFAULT_IMAGE_TOKEN}\n{row['question']}" if not no_image else row['question']
                },
                {
                    "from": "gpt",
                    "value": row["answer"]
                }
            ],
            "subject_id": row["subject_id"],
            "study_id": row["study_id"],
            "image_id": row["image_id"],
            "semantic_type": row["semantic_type"],
            "content_type": row["content_type"],
            "answer_type": row["answer_type"],
            "gender": row["gender"],
            "age": row["age"],
            "year": row["year"],
            "ethnicity": row["ethnicity"]
        }
        final_data.append(new_entry)

    with open(str(save_path), 'w') as output_file:
        json.dump(final_data, output_file, indent=4)


def get_json_filename(data_dir:Path, ood_value:str,
                      test_folder_name:str, train_folder_name:str,
                      val_folder_name:str,mod:str,dataset_name:str, 
                      split:str, data_shift:str, no_image:bool = False):
    ''' 
    # TODO: add explanation here
    dataset_name: This is the name of the file you want to load (train, test, val)

    This function implements the data shift to the given dataset
    
        Inputs:
                data_dir (Path): Path to the dataset directory
                output_file_name (str): File name to store the samples after data shift
                ood_value (str): The values defining the out-of-distribution samples. Samples having this value are assigned as ood samples, rest as iid
                dataset_name (str): Name of the dataset for the experiment
                mod (str): The mod of the experiment (values: train, val, test)
                ...


        Outputs:
                path to the output file
    '''

    if split == "all" or split == "sample":
        split_category = None
        split_value = None
        output_file_name = f"{dataset_name}_{mod}_{split}".replace(" ", "")
    else:
        split_category = data_shift
        split_value = ood_value
        output_file_name = f"{dataset_name}_{mod}_{split}_{split_category}_{split_value}".replace(" ", "")

    if no_image:
        output_file_name = output_file_name + '_no_image'

    if not os.path.isdir(data_dir / "split_files"):
        os.makedirs(data_dir / "split_files")

    if os.path.isfile(data_dir / "split_files" / f"{output_file_name}.json"):
        return data_dir / "split_files" / f"{output_file_name}.json", output_file_name
    else:
        if dataset_name == "SLAKE":
            df = get_slake_df(data_dir=data_dir, test_folder_name=test_folder_name,
                              train_folder_name=train_folder_name,val_folder_name=val_folder_name,mod=mod, 
                              split=split, split_category=split_category, split_value=split_value)
            convert_raw_to_final(df, data_dir / "split_files" / f"{output_file_name}.json", no_image=no_image)
        elif dataset_name == "OVQA":
            if split_value is not None:
                split_value = split_value.replace("_", " ")
            df = get_ovqa_df(data_dir=data_dir, test_folder_name=test_folder_name,
                              train_folder_name=train_folder_name,val_folder_name=val_folder_name,mod=mod, 
                              split=split, split_category=split_category, split_value=split_value)
            convert_ovqa_raw_to_final(df, data_dir / "split_files" / f"{output_file_name}.json", no_image=no_image)
        elif dataset_name == "LIDC":
            df = get_lidc_df(data_dir=data_dir, test_folder_name=test_folder_name,
                              train_folder_name=train_folder_name, val_folder_name=val_folder_name, mod=mod,
                              split=split, split_category=split_category, split_value=split_value)
            convert_lidc_raw_to_final(df, data_dir / "split_files" / f"{output_file_name}.json", no_image=no_image)
        elif dataset_name == "MIMIC":
            df = get_mimic_df(data_dir=data_dir, test_folder_name=test_folder_name,
                              train_folder_name=train_folder_name, val_folder_name=val_folder_name, mod=mod,
                              split=split, split_category=split_category, split_value=split_value)
            convert_mimic_raw_to_final(df, data_dir / "split_files" / f"{output_file_name}.json", no_image=no_image)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented")

        return data_dir / "split_files" / f"{output_file_name}.json", output_file_name
    
# def get_json_filename(data_dir:Path, name:str):
#     identifier = name.split("_")
#     dataset = identifier[0]
#     if not os.path.isdir(data_dir / "split_files"):
#         os.makedirs(data_dir / "split_files")

#     if os.path.isfile(data_dir / "split_files" / f"{name}.json"):
#         return data_dir / "split_files" / f"{name}.json"
#     else:
#         mode = identifier[1]
#         split = identifier[2]
#         if split != "all":
#             # this is not good but for now lets keep it
#             if dataset=="SLAKE":
#                 # TODO: fix the issues here "-" "_" for split valie
#                 split_category = identifier[3].replace("-", "_")
#                 split_value = identifier[4]
#             elif dataset == "OVQA":
#                 # TODO: fix the issues here "-" "_" for split valie
#                 split_category = identifier[3].replace("-", "_")
#                 split_value = identifier[4].replace("-", "_") 
#             else:
#                 raise NotImplementedError(f"Dataset {dataset} not implemented")
#         else:
#             split_category = None
#             split_value = None
#         if dataset == "SLAKE":
#             df = get_slake_df(data_dir=data_dir, mode=mode, split=split, split_category=split_category,
#                               split_value=split_value)
#             convert_raw_to_final(df, data_dir / "split_files" / f"{name}.json")
#         elif dataset == "OVQA":
#             df = get_ovqa_df(data_dir=data_dir, mode=mode, split=split, split_category=split_category,
#                              split_value=split_value)
#             convert_ovqa_raw_to_final(df, data_dir / "split_files" / f"{name}.json")
#         else:
#             raise NotImplementedError(f"Dataset {dataset} not implemented")

#         return data_dir / "split_files" / f"{name}.json"
