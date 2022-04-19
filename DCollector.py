#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  27 16:30:21 2019

@author: kennedy
"""
import os
import datetime
from oandapyV20 import API


#--Create and append folders
class Path(object):
    def __init__(self, path):
        self.path = path
        self.loadPath()
        
    def loadPath(self):
      import os
      try:
        if os.path.exists(self.path['mainPath']):
          try:
            FOLDERS = ['\\DATASETS', 
                       '\\PREDICTED', 
                       '\\IMAGES',
                       '\\TICKERS',
                       '\\MODEL',
                       '\\SIGNALS']
            FOLDER_COUNT = 0
            for folders in FOLDERS:
              '''If folder is not created or created but deleted..Recreate/Create the folder.
              Check for all folders in the FOLDERS list'''
              if not os.path.exists(self.path['mainPath'] + FOLDERS[FOLDER_COUNT]):
                os.makedirs(self.path['mainPath'] + FOLDERS[FOLDER_COUNT])
                print('====== 100% Completed ==== : {}'.format(self.path['mainPath'] + FOLDERS[FOLDER_COUNT]))
                FOLDER_COUNT += 1
              elif os.path.exists(self.path['mainPath'] + FOLDERS[FOLDER_COUNT]):
                '''OR check if the file is already existing using a boolean..if true return'''
                print('File Already Existing : {}'.format(self.path['mainPath'] + FOLDERS[FOLDER_COUNT]))
                FOLDER_COUNT += 1
          except OSError as e:
              '''raise OSError('File Already Existing {}'.format(e))'''
              print('File Already existing: {}'.format(e))
        elif not os.path.exists(self.path['mainPath']):
            raise OSError('File self.path: {} does not exist\n\t\tPlease check the self.path again'.format(self.path['mainPath']))
        else:
            print('File Already Existing')
      except Exception as e:
          raise(e)
      finally:
          print('Process completed...Exiting')

#--download and append stocks/instruments to folder
class stockDownload:
    def __init__(self, path, instrument, start, end, client, timeframe):
        self.path = path
        self.instrument = instrument
        self.start = start
        self.end = end
        self.client = client
        self.timeframe = timeframe
        self.downloadStockData()
        
    def downloadStockData(self):
        '''
          :Arguments:
            :instruments:
              Name of the instrument we are trading
            :start: specify the start date of stcok to download
            :end: specify end date of the stock to download
            
          :Returntype:
            return the csv file of the downloaded stock in the
            specific folder.
        '''
        from oandapyV20.contrib.factories import InstrumentsCandlesFactory
        def covert_json(reqst, frame):
            for candle in reqst.get('candles'):
                ctime = candle.get('time')[0:19]
                try:
                    rec = '{time},{complete},{o},{h},{l},{c},{v}'.format(time = ctime,
                           complete = candle['complete'],
                           o = candle['mid']['o'],
                           h = candle['mid']['h'],
                           l = candle['mid']['l'],
                           c = candle['mid']['c'],
                           v = candle['volume'])
                except Exception as e:
                    raise(e)
                else:
                    # frame.write('datetime, completed, open, high, low, close, volume')
                    frame.write(rec+'\n')

                
        #try except to both create folder and enter ticker
        try:
            #create folder for all instruments
            if not os.path.exists(self.path['mainPath'] + f'\\DATASETS\\{self.instrument}'):
                os.makedirs(self.path['mainPath'] + f'\\DATASETS\\{self.instrument}')
            #import the required timeframe
            header = "datetime, completed, open, high, low, close, volume"
            # with open(self.path['mainPath'] + '\\DATASETS\\{}\\{}_{}.csv'.format(self.instrument, self.instrument,self.timeframe), 'w') as f:
            #     f.write(header + "\n")
            #     f.close()
            with open(self.path['mainPath'] + '\\DATASETS\\{}\\{}_{}.csv'.format(self.instrument, self.instrument, self.timeframe), 'w') as OUTPUT:
                # OUTPUT.write(header + "\n")
                params = {'from': self.start,
                          'to': self.end,
                          'granularity': self.timeframe,
                          }
                try:
                  for ii in InstrumentsCandlesFactory(instrument = self.instrument, params = params):
                      print("REQUEST: {} {} {}".format(ii, ii.__class__.__name__, ii.params))
                      self.client.request(ii)
                      covert_json(ii.response, OUTPUT)
                      print(InstrumentsCandlesFactory, OUTPUT)
                except Exception as e:
                    from oandapyV20.exceptions import V20Error # Most likely required
                    error = V20Error(code, msg) # oandapyV20.V20Error(code, msg)
                    print ('Oanda Error: ', error)
                    print('{} not available. \n Please check your internet connection'.format(self.instrument))
                print('********************Data-set download process complete******************\n{}_{}\n'.format(self.instrument, self.timeframe))
        except Exception as e:
            raise(e)
        finally:
            print('*'*40)
            print('Stock download completed')
            print('*'*40)

class Runcollector:
    def __init__(self, path, start, end, client, timeframe):
        self.path = path
        self.start = start
        self.end = end
        self.client = client
        self.timeframe = timeframe
        self.runnewMain()

    def loadData(self):
        from threading import Thread
        threads = []
        for instr in self.path['instruments'].split(','):
            threads.append(Thread(target = stockDownload, args = (self.path, instr, self.start,
                                                                                   self.end, self.client, self.timeframe)))
        for trd in threads:
            trd.daemon = True
            trd.start()
        for st_trd in threads:
            st_trd.join()
                
    def runnewMain(self):
        import time
        return self.loadData()


# if __name__ == '__main__':
#     current_directory = os.getcwd()
#
#     path = {'mainPath': str(current_directory),
#             'acountPath': 'DOCS\\account_id.txt',
#             'tokenPath': 'DOCS\\token_live.txt',
#             'tokenPath_pract': 'DOCS\\token_pract.txt',
#             'telegram': 'DOCS\\telegram.txt',
#             'predicted': 'PREDICTED',
#             'signals': 'SIGNALS',
#             'start': str((datetime.datetime.utcnow() - datetime.timedelta(days=730)).isoformat('T')[:-7] + 'Z'),
#             # '2019-10-03T00:00:00Z', # I changed this from match to oct
#             'end': str(datetime.datetime.utcnow().isoformat('T')[:-7] + 'Z'),
#             'environment': 'live',
#             'strategy': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11',
#                          '22', '33', '44', '55', '66', '77', '88', '99', '111',
#                          '222', '333', '444', '555', '666', '777', '888', '999', '1111',
#                          '2222', '3333', '4444', '5555', '6666', '7777', '8888'],
#             'instruments': 'USD_CAD',
#             # 'instruments': 'AUD_USD,BCO_USD,BTC_USD,DE30_EUR,EUR_AUD,EUR_JPY,EUR_USD,GBP_JPY,GBP_USD,NAS100_USD,SPX500_USD,US30_USD,USD_CAD,USD_JPY,XAU_USD',
#             'timeframes': ['M15', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8',
#                            'H12', 'D', 'W']}
#
#     if path['environment'] == 'live':
#         with open(os.path.join(path['mainPath'], path['tokenPath'])) as tk:
#             token = tk.readline().strip()
#             client = API(access_token=token, environment=path['environment'])
#     else:
#         with open(os.path.join(path['mainPath'], path['tokenPath_pract'])) as tk:
#             token = tk.readline().strip()
#             client = API(access_token=token, environment=path['environment'])
#
#     Path(path)
#     Runcollector(path, path['start'], path['end'], client, 'M1')
