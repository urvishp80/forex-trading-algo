# forex-traading-algo
ML Model based on trading strategy for forex trading

## Steps to run the repo:-
1) Installation
  -	First download the python by following the steps from the pdf.
  -	Then clone the repo.  
  -	After that create a new environment by hitting a command (conda create -n forex-trading python=3.8)
  -	Then activate this environment by hitting a command (conda activate forex-trading)
  -	After that install all the requirements by (pip install –r requirements.txt) which is already present in the repo.
  -	If problem comes in installing TA-LIB then install it by pip install TA_Lib-0.4.24-cp38-cp38-win_amd64.whl
  -	ALL SET!
 2) Dataset Collection for training
   - Automated_Signal_generator.py for collecting the data of all strategy you can change time and timeframes from there.
   - merge_all_coin.py for combining all the results of strategy.
   - max_count.py for getiing final labels by max-voting all the strategy results.
   - split_dataset.py for splitting the dataset in training and unseen dataset you can change the {Date} parameter there according to dataset collected.
   - Done!
 3) Training
   - config.py in that you have change the BASE_PATH you have to put your base-path there where you have clone this repo.
   - main.py for training the model.
   - Done your model will be saved in weights folder.
 4) Inference
   - You can update yours telegram credential in DOCS/telegram.txt
   - automated_gui_ml.py for getting GUI + Telegram messages in that you can change the parameter like (time, timeframe and currency)
   -  automated_gui.py for running old gui 
