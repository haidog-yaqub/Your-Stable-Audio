import concurrent.futures
import pandas as pd
from tqdm import tqdm
from ytb_utils import download_audioset_file

import warnings
warnings.filterwarnings("ignore")


def download_wrapper(row, meta_name, out_path):
    audiocap_id = row['audiocap_id']
    youtube_id = row['youtube_id']
    start_time = row['start_time']
    end_time = start_time + 10
    msg = download_audioset_file(youtube_id, start_time, end_time, out_path + meta_name, file_name=str(audiocap_id))
    return msg


if __name__=='__main__':
    for meta_name in ['train', 'val', 'test']:
        out_path = 'audiocaps_48k/'
        df = pd.read_csv(meta_name+'.csv')
    
        # List to hold messages from each download
        messages = []
        
        # Define the number of threads
        num_threads = 4  # You can adjust this number based on your needs
        total_tasks = len(df)  # Total number of tasks to process

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Prepare a list to store futures
            futures = [executor.submit(download_wrapper, row, meta_name, out_path) for index, row in df.iterrows()]

            # Process as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, unit="file"):
                try:
                    msg = future.result()
                    messages.append(msg)
                    # Process the message as needed
                    # ...
                except Exception as exc:
                    print('A task generated an exception: %s' % exc)

        messages = pd.DataFrame(messages, columns=['filename','msg'])
        messages.to_csv(out_path+meta_name+'_msg.csv', index=False)
