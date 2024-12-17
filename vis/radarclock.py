""" Tadam """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from PIL import Image
import glob
import os

def craft_radar_plot(data_file, img_folder):
    """ """

    # parameters
    day_start = 12
    day_end = 23
    night_start = 0
    night_end = 11
    
    df = pd.read_csv(data_file)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    event_counts = df['hour'].value_counts().sort_index()
    event_counts_dict = event_counts.to_dict()

    # extract max value
    max_nb_of_event = max(event_counts_dict, key=event_counts_dict.get)

    # generate day part
    for x in range(day_start, day_end+1):
        
        key_to_keep = []
        for h in event_counts_dict:
            if h >= day_start and h <= x:
                key_to_keep.append(h)

        data = {}
        for k in range(day_start, day_end+1):
            if k in key_to_keep:
                data[str(k)] = event_counts_dict[k]
            else:
                data[str(k)] = 0
                
        # craft plot
        craft_radar(data, f"{img_folder}/day_{x}.png", max_nb_of_event)
        
    # generate night part
    for x in range(night_start, night_end+1):
        
        key_to_keep = []
        for h in event_counts_dict:
            if h >= night_start and h <= x:
                key_to_keep.append(h)

        data = {}
        for k in range(night_start, night_end+1):
            if k in key_to_keep:
                data[str(k)] = event_counts_dict[k]
            else:
                data[str(k)] = 0
                
        # craft plot
        craft_radar(data, f"{img_folder}/night_{x}.png", max_nb_of_event)




def craft_radar(hour_to_event:dict, img_name:str, max_value:int):
    """Keys must be string, values must be integer"""

    # TODO : set max value
    df = pd.DataFrame(
        dict(
            r=list(hour_to_event.values()),
            theta=list(hour_to_event.keys()),
        )
    )
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_value]
            )
        ),
        showlegend=False
    )
    fig.write_image(img_name, width=1920, height=1080)





def generate_gif(img_folder:str, day_start_frame:int, day_stop_frame:int, night_start_frame:int, night_stop_frame:int, output_name:str):
    """ """

    # Create the frames for day
    frames = []
    for x in range(day_start_frame, day_stop_frame+1):
        img_file_name = f"{img_folder}/day_{x}.png"
        if(os.path.isfile(img_file_name)):
            new_frame = Image.open(img_file_name)
            frames.append(new_frame)

    # add night frames
    for x in range(night_start_frame, night_stop_frame+1):
        img_file_name = f"{img_folder}/night_{x}.png"
        if(os.path.isfile(img_file_name)):
            new_frame = Image.open(img_file_name)
            frames.append(new_frame)
            
    # Save into a GIF file that loops forever
    frames[0].save(
        output_name,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=300,
        loop=0
    )




def plot_a_day(df, day):
    """ """

    # extract infos for the given day

    # generate radar plots

    # generate gif
    





if __name__ == "__main__":



    # parameters
    data_file = 'data/events.csv'
    data_folder = '/tmp/radar'
    hour_to_event = {
        '12':4,
        '13':5,
        '14':7,
        '15':2,
        '16':3,
        '17':9,
        '18':8,
        '19':5,
        '20':5,
        '21':0,
        '22':1,
        '23':3
    }

    # craft_radar(hour_to_event)



    craft_radar_plot(data_file, data_folder)
    generate_gif(
        data_folder,
        12,
        23,
        0,
        11,
        "test.gif"
    )