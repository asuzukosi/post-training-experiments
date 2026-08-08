import os
import random
import pandas as pd

DATA_DIRECTORY = "data/"

# Translation map used to translate condition column data from german to english
TRANSLATION_MAP = {
    "Stau": "Traffic",
    "2 Normal": "Normal",
    "Normal": "Normal",
    "Stau Messfehler": "Traffic Jam Measurement error",
    "Frei": "Free",
    "Frei Vollbremsung": "Emergency Braking",
    "Normal Glatteis": "Normal Icy Road",
    "Frei Beschleunigung": "Free Accelaration",
    
}



def load_dataset(directory):
    # read all the files from the data directory
    files = os.listdir(directory)
    number_of_files = len(files)

    file = files[random.randint(0, 10)]
    keys = file.split(".")[0].split("_")
    
    df = None

    for file in files:
        if file.endswith(".csv"):
            # extract key data
            keys = file.split(".")[0].split("_")
            date = keys[0]
            brand = keys[1]
            model = keys[2]
            source = keys[3]
            destination = keys[4]
            if(len(keys) == 6):
                condition = TRANSLATION_MAP[keys[5]]
            elif (len(keys)==7):
                condition = TRANSLATION_MAP[" ".join([keys[5], keys[6]])]
            else:
                raise Exception(f"Keys from file exceded amounnt that was sheduled to be processed \n keys: {keys} \n number of keys {len(keys)} \n We can only process 6 or 7 keys")

            # read file and create dataframe
            if df is not None:
                new_df = pd.read_csv(os.path.join(directory, file))
                new_df["date"] = date
                new_df["brand"] = brand
                new_df["model"] = model
                new_df["source"] = source
                new_df["destination"] = destination
                new_df["condition"] = condition
                df = pd.concat([df, new_df])
            else:
                df = pd.read_csv(os.path.join(directory, file))
                df["date"] = date
                df["brand"] = brand
                df["model"] = model
                df["source"] = source
                df["destination"] = destination
                df["condition"] = condition
        else:
            continue
    return df
    
    

def postprocess_dataset(df: pd.DataFrame):
    df.drop_duplicates(subset=['Time', 'model', 'date'], keep='first', inplace=True)
    df = df.dropna(subset=['Accelerator Pedal Position E [%]'])
    df.drop(columns=['Engine Coolant Temperature [Â°C]', 'Intake Air Temperature [Â°C]',
       'Ambient Air Temperature [Â°C]'], inplace=True)
    df['Time'] = pd.to_datetime(df['Time'])
    unique_days = df['date'].unique()
    new_dfs = []
    for day in unique_days:
        # get all the unique pairs of source, destination and condition for a specific day
        sub_df: pd.DataFrame = df[df['date'] == day]
        unique_pairs = sub_df[['source', 'destination', 'condition']].drop_duplicates()
        for index, row in unique_pairs.iterrows():
            # create sub sub df based on the source, destination and condition
            sub_sub_df: pd.DataFrame = sub_df[(sub_df["source"] == row["source"]) & (sub_df["destination"] == row["destination"]) & (sub_df["condition"] == row["condition"])]
            # old_size = sub_sub_df.size
            
            # Round timestamps down to the nearest 10-minute interval
            sub_sub_df['RoundedTime'] = sub_sub_df['Time'].dt.floor('10min')
            
            # group sub sub df by the rounded time
            sub_sub_df = sub_sub_df.groupby('RoundedTime').first()
            # new_size = sub_sub_df.size
            
            # calculate the amount of data dropped on the sub sub df
            # print("Amount reduced by : ", old_size - new_size)
            
            # append to new_dfs list
            new_dfs.append(sub_sub_df)
    df = pd.concat(new_dfs)
    df['date'] = pd.to_datetime(df['date'])
    return df